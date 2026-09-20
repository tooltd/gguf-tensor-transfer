"""GGUF Tensor Transfer GUI — A (donor) -> B (base) -> C (new GGUF)

Replace selected tensors in base B with same-named tensors from donor A,
streaming ~13GB without loading tensors into RAM.

The table lists every tensor of B in order, then any A-only tensors (e.g. an
MTP block blk.64 missing from B) so they can be selected and appended to C.
C metadata is synced from A for structural KVs: {arch}.block_count is raised
and {arch}.nextn_predict_layers is patched/appended when needed (llama.cpp
requires it to activate the MTP layer).

Core logic (parse/plan/write) is importable & GUI-free; tkinter UI at bottom.
"""
from __future__ import annotations
import os, sys, time, threading, queue, struct

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _gguf_fast as G
from gguf.constants import GGML_QUANT_SIZES

# 256MB chunks: ~2x faster than 64MB on contiguous regions of the user's SATA HDD
CHUNK = 256 * 1024 * 1024
PIPE_BUF = 2  # chunks in flight between reader and writer threads
PROG_STEP = 32 * 1024 * 1024  # progress/cancel granularity while writing

# donor KVs (under general.architecture) synced into C when appending donor-only tensors
KV_FIX_SUFFIXES = ('block_count', 'nextn_predict_layers')


def fmt(n: int) -> str:
    for u in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024 or u == 'TB':
            return f'{int(n):,} B' if u == 'B' else f'{n:,.1f} {u}'
        n /= 1024


def fmt_delta(n: int) -> str:
    if n == 0:
        return '±0'
    return ('+' if n > 0 else '−') + fmt(abs(n))


def fmt_dims(dims: list) -> str:
    return 'x'.join(str(d) for d in dims)


class TensorSpec:
    __slots__ = ('name', 'dims', 'type', 'src_off', 'size')
    def __init__(self, name, dims, ttype, src_off, size):
        self.name, self.dims, self.type = name, dims, ttype
        self.src_off, self.size = src_off, size  # absolute offset in source file
    @property
    def type_name(self): return G.type_name(self.type)
    @property
    def tbytes(self): return f'{fmt(self.size)}'


def parse_gguf(path: str) -> dict:
    """Fast struct-only parse. Returns dict with kv, tensors (TensorSpec, absolute offs), ..."""
    f = open(path, 'rb')
    try:
        h = G.parse_header(f)
    finally:
        f.close()
    kv = {k.key: k.value for k in h.kv}
    data_off = h.data_offset
    tensors = []
    for t in h.tensors:
        ne = 1
        for d in t.dims: ne *= d
        elems, ts = GGML_QUANT_SIZES[t.type]
        size = ne * ts // elems
        tensors.append(TensorSpec(t.name, list(t.dims), t.type, data_off + t.offset, size))
    return {
        'path': path, 'kv': kv, 'kv_list': h.kv, 'kv_raw_range': h.kv_raw_range,
        'tensors': tensors, 'by_name': {t.name: t for t in tensors},
        'data_offset': data_off, 'alignment': h.alignment,
        'kv_count': len(h.kv),
        'file_size': os.path.getsize(path),
    }


class Plan:
    def __init__(self, base: dict, src: dict, selected: set):
        self.base, self.src = base, src
        self.selected = selected
        self.warnings = []
        self.out_tensors = []   # C tensor list (TensorSpec with src file tag via tuple)
        self.extra = []         # (TensorSpec) donor-only tensors appended
        self.kv_patches = []    # human-readable metadata changes applied to C
        self.kv_patch_vals = {} # key -> new value (patched in place in B's KV section)
        self.kv_append = []     # (key, G.KV) appended to C's KV section
        self.build()

    def build(self):
        b, s = self.base, self.src
        smap = s['by_name']
        off = 0
        # B's tensors in order
        for t in b['tensors']:
            if t.name in self.selected:
                st = smap.get(t.name)
                if st is None:
                    self.warnings.append(f'{t.name}: missing in donor, kept from base')
                    self.out_tensors.append(('B', t, off)); off += t.size
                    continue
                if st.dims != t.dims:
                    self.warnings.append(f'{t.name}: shape mismatch {fmt_dims(t.dims)} vs {fmt_dims(st.dims)}, kept from base')
                    self.out_tensors.append(('B', t, off)); off += t.size
                    continue
                self.out_tensors.append(('A', st, off))
                off += st.size
            else:
                self.out_tensors.append(('B', t, off))
                off += t.size
            off += (-off) % 8
        # donor-only tensors (not in base) — appended
        bnames = {t.name for t in b['tensors']}
        for name in sorted(self.selected):
            if name not in bnames and name in smap:
                st = smap[name]
                self.out_tensors.append(('A', st, off))
                self.extra.append(st)
                off += st.size
                off += (-off) % 8
        self._compute_kv_fixes()
        self.total_data = off
        # coalesce into contiguous spans per source file: (src_tag, src_start, length)
        # NOTE: merging is only valid if the tensors are ALSO contiguous (same order)
        # in the source file. Files can store same-named tensors in different orders
        # (e.g. MTP block), so every merged tensor must sit at its expected source offset.
        self.spans = []
        i = 0
        while i < len(self.out_tensors):
            tag, t, o = self.out_tensors[i]
            end = o + t.size  # C-side end
            j = i + 1
            while j < len(self.out_tensors):
                tag2, t2, o2 = self.out_tensors[j]
                if tag2 != tag or o2 != end:
                    break
                if t2.src_off != t.src_off + (o2 - o):  # not contiguous in source
                    self.warnings.append(f'{t2.name}: donor stores it in a different order '
                                         f'than base — copied separately (no coalescing)')
                    break
                end = o2 + t2.size
                j += 1
            self.spans.append((tag, t.src_off, end - o))
            i = j

    def _compute_kv_fixes(self):
        """Structural metadata to sync into C when donor-only tensors are appended."""
        if not self.extra:
            return
        arch = self.base['kv'].get('general.architecture')
        if not arch:
            return
        for suf in KV_FIX_SUFFIXES:
            key = f'{arch}.{suf}'
            skv = next((k for k in self.src['kv_list'] if k.key == key), None)
            if skv is None or skv.vtype not in G._SCALAR_FMT:
                continue
            sval = skv.value
            bval = self.base['kv'].get(key)
            if bval is None:
                self.kv_append.append((key, skv))
                self.kv_patches.append(f'+ {key} = {sval}')
            elif isinstance(bval, int) and isinstance(sval, int) and sval > bval:
                self.kv_patch_vals[key] = sval
                self.kv_patches.append(f'{key}: {bval} → {sval}')

    def _kv_append_bytes(self) -> bytes:
        out = b''
        for key, kv in self.kv_append:
            out += G.pack_kv(kv)
        return out

    def header_len(self) -> int:
        """Exact C header size without disk I/O (for live previews)."""
        b = self.base
        n = 8 + (b['kv_raw_range'][1] - b['kv_raw_range'][0]) + len(self._kv_append_bytes())
        for tag, t, o in self.out_tensors:
            n += 8 + len(t.name.encode('utf-8')) + 4 + 8 * len(t.dims) + 4 + 8
        align = b['alignment']
        return n + (-n) % align

    def data_off_c(self) -> int:
        return len(self.header_bytes())

    def header_bytes(self) -> bytes:
        b, s = self.base, self.src
        ks, ke = b['kv_raw_range']
        fb = open(b['path'], 'rb')
        fb.seek(ks)
        kv_raw = bytearray(fb.read(ke - ks))
        fb.close()
        for key, val in self.kv_patch_vals.items():
            self._patch_kv_inplace(kv_raw, key, val)
        if self.kv_append:
            kv_raw += self._kv_append_bytes()
        entries = [G.TensorInfo(t.name, len(t.dims), t.dims, t.type, o) for tag, t, o in self.out_tensors]
        out = struct.pack('<IIQQ', G.MAGIC, 3, len(entries), b['kv_count'] + len(self.kv_append))
        out += bytes(kv_raw)
        for e in entries:
            out += G.pack_tensor_entry(e)
        align = b['alignment']
        pad = (-len(out)) % align
        return out + b'\0' * pad

    @staticmethod
    def _patch_kv_inplace(kv_raw: bytearray, key: str, value) -> None:
        kb = key.encode('utf-8')
        i = kv_raw.find(kb)
        if i < 0 or i < 8:
            return
        if struct.unpack_from('<Q', kv_raw, i - 8)[0] != len(kb):  # not a real key boundary
            return
        j = i + len(kb)
        vtype = struct.unpack_from('<I', kv_raw, j)[0]
        fc = G._SCALAR_FMT.get(vtype)
        if fc is None:
            return
        nb = struct.pack(fc, value)
        kv_raw[j + 4:j + 4 + len(nb)] = nb


def write_output(plan: Plan, out_path: str, progress_cb=None, cancel=None) -> None:
    """Stream C = header + spans. Reader thread feeds a bounded queue to the
    writer (this thread) so disk reads and writes overlap. Each 256MB chunk is
    written in 32MB sub-steps so progress/cancel tick every ~0.3-0.7s, not per
    chunk. A 'started' tick fires right after the header is written."""
    header = plan.header_bytes()
    fb = open(plan.base['path'], 'rb')
    fa = open(plan.src['path'], 'rb')
    total = len(header) + plan.total_data
    done = 0
    t0 = time.time()

    def reader(q, err_q):
        try:
            for tag, soff, length in plan.spans:
                fs = fb if tag == 'B' else fa
                fs.seek(soff)
                rem = length
                while rem > 0:
                    if cancel is not None and cancel():
                        raise InterruptedError('cancelled')
                    n = min(CHUNK, rem)
                    buf = fs.read(n)
                    if not buf:
                        raise IOError(f'short read in source {fs.name} at {soff + rem - n}')
                    q.put(buf)
                    rem -= len(buf)
        except Exception as e:
            err_q.put(e)
        finally:
            q.put(None)  # sentinel — always unblocks the writer

    q = queue.Queue(maxsize=PIPE_BUF)
    err_q = queue.Queue()
    th = threading.Thread(target=reader, args=(q, err_q), daemon=True)
    try:
        th.start()
        with open(out_path, 'wb') as fo:
            fo.write(header)
            done += len(header)
            if progress_cb:  # 'started' tick — bar/speed label appear immediately
                progress_cb(done, total, time.time() - t0)
            while True:
                buf = q.get()
                if buf is None:
                    break
                if cancel is not None and cancel():
                    raise InterruptedError('cancelled')
                if err_q.qsize():
                    raise err_q.get()
                view = memoryview(buf)
                for off in range(0, len(buf), PROG_STEP):
                    if cancel is not None and cancel():
                        raise InterruptedError('cancelled')
                    n = min(PROG_STEP, len(buf) - off)
                    fo.write(view[off:off + n])
                    done += n
                    if progress_cb:
                        progress_cb(done, total, time.time() - t0)
        if err_q.qsize():
            raise err_q.get()
    except:
        try: os.remove(out_path)
        except OSError: pass
        raise
    finally:
        th.join()
        fb.close(); fa.close()


def extract_donor(src: dict, names: set, out_path: str, arch: str) -> int:
    """Write a small GGUF containing only `names` from src. Returns tensor count."""
    sel = [t for t in src['tensors'] if t.name in names]  # keep source order
    if not sel:
        raise ValueError('nothing to extract')
    kv = [G.KV('general.architecture', G.V_STR, arch),
          G.KV('general.name', G.V_STR, f'donor-{len(sel)}-tensors')]
    # keep structural KVs so a Plan using this donor can sync them into C
    for k in src['kv_list']:
        if k.key == f'{arch}.block_count' or k.key == f'{arch}.nextn_predict_layers':
            kv.append(k)
    entries, off = [], 0
    for t in sel:
        entries.append((G.TensorInfo(t.name, len(t.dims), t.dims, t.type, 0), off, t))
        off += t.size
        off += (-off) % 8
    for e, o, _t in entries:  # offset relative to data start == data_offset
        e.offset = o
    header = G.build_header_bytes(kv, [e for e, _, _ in entries], 32)  # GGUF default alignment
    with open(out_path, 'wb') as fo:
        fo.write(header)
        for e, o, t in entries:
            f = open(src['path'], 'rb')
            f.seek(t.src_off)
            rem = t.size
            while rem > 0:
                n = min(CHUNK, rem)
                fo.write(f.read(n))
                rem -= n
            f.close()
    return len(sel)


def verify_output(plan: Plan, out_path: str) -> list:
    """Light verification: header parse, sizes, spot byte-compare of selected tensors."""
    errs = []
    f = open(out_path, 'rb')
    try:
        h = G.parse_header(f)
    except Exception as e:
        return [f'cannot parse output header: {e}']
    finally:
        f.close()
    if len(h.tensors) != len(plan.out_tensors):
        errs.append(f'tensor count {len(h.tensors)} != {len(plan.out_tensors)}')
    fsz = os.path.getsize(out_path)
    need = len(plan.header_bytes()) + plan.total_data
    if fsz != need:
        errs.append(f'file size {fsz:,} != expected {need:,}')
    # spot check: first 1KB of each donor-sourced tensor
    checks = 0
    for tag, t, o in plan.out_tensors:
        if tag != 'A':
            continue
        checks += 1
        off_c = plan.data_off_c() + o
        fo = open(out_path, 'rb'); fs = open(plan.src['path'], 'rb')
        n = min(1024, t.size)
        fo.seek(off_c); aa = fo.read(n)
        fs.seek(t.src_off); bb = fs.read(n)
        fo.close(); fs.close()
        if aa != bb:
            errs.append(f'{t.name}: data mismatch at C offset {off_c:,}')
    if not errs:
        errs.append(f'OK: {len(h.tensors)} tensors, {checks} donor tensors spot-checked, size {fsz:,}')
    return errs


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
def _main_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog

    class TRow:
        """One table row: tensor name with its TensorSpec in A (donor) and/or B (base)."""
        __slots__ = ('name', 'a', 'b')
        def __init__(self, name, a, b):
            self.name, self.a, self.b = name, a, b

    class App:
        COLS = ('sel', 'name', 'a', 'b', 'c', 'delta')
        COL_W = {'sel': 38, 'name': 420, 'a': 190, 'b': 190, 'c': 205, 'delta': 115}
        COL_L = {'sel': '✓', 'name': 'tensor', 'a': 'A · donor', 'b': 'B · base',
                 'c': 'C · result', 'delta': 'Δ size (B → C)'}

        def __init__(self, root: tk.Tk):
            self.root = root
            self.q: queue.Queue = queue.Queue()
            self.cancel_ev = threading.Event()
            self.busy = False
            root.title('tooltd · GGUF Tensor Transfer  —  A (donor) → B (base) → C (result)')
            root.geometry('1380x900')
            self._theme()
            self.parsed = {}    # role -> parse dict
            self.role_path = {'A': tk.StringVar(), 'B': tk.StringVar(), 'C': tk.StringVar()}
            self.role_file = {}
            self.sel = set()
            self.rows: list[TRow] = []
            self.sort_col = 'name'
            self.sort_dir = 1
            self._build_ui()
            self.root.after(100, self._poll)

        # ---------- theme ----------
        def _theme(self):
            s = ttk.Style(self.root)
            try:
                s.theme_use('clam')
            except tk.TclError:
                pass
            bg, card, field, head = '#14151a', '#1d2028', '#0f1116', '#262a37'
            fg, dim = '#e6e8ef', '#9095a8'
            self.root.configure(bg=bg)
            s.configure('.', background=bg, foreground=fg, font=('Segoe UI', 9))
            s.configure('TFrame', background=bg)
            s.configure('Card.TFrame', background=card)
            s.configure('TLabelframe', background=card, foreground='#8b93b0', borderwidth=1, relief='solid')
            s.configure('TLabelframe.Label', background=card, foreground='#aab2d0', font=('Segoe UI', 9, 'bold'))
            s.configure('TLabel', background=card, foreground=fg)
            s.configure('Dim.TLabel', background=card, foreground=dim)
            s.configure('TButton', background='#2b3040', foreground=fg, borderwidth=0, padding=(10, 4))
            s.map('TButton', background=[('active', '#39415a'), ('disabled', '#22252f')],
                  foreground=[('disabled', '#5c6274')])
            s.configure('Primary.TButton', background='#2563eb', foreground='#fff', font=('Segoe UI', 10, 'bold'), padding=(14, 6))
            s.map('Primary.TButton', background=[('active', '#3b82f6')])
            s.configure('Success.TButton', background='#059669', foreground='#fff', font=('Segoe UI', 10, 'bold'), padding=(14, 6))
            s.map('Success.TButton', background=[('active', '#10b981')])
            s.configure('TEntry', fieldbackground=field, foreground=fg, insertcolor=fg, borderwidth=0, padding=3)
            s.configure('Treeview', background=field, fieldbackground=field, foreground=fg,
                        font=('Consolas', 9), rowheight=21, borderwidth=0)
            s.configure('Treeview.Heading', background=head, foreground='#aab2d0', font=('Segoe UI', 9, 'bold'), padding=5)
            s.map('Treeview.Heading', background=[('active', '#333a52'), ('pressed', '#3d4560')])
            s.map('Treeview', background=[('selected', '#274690')], foreground=[('selected', '#ffffff')])
            s.configure('.Horizontal.TProgressbar', background='#2563eb', troughcolor='#22252f', borderwidth=0)
            self._field_bg = field

        # ---------- ui ----------
        def _build_ui(self):
            r = self.root

            # ---- files card ----
            top = ttk.LabelFrame(r, text='  FILES   A (donor) → B (base) → C (result)  ', padding=(10, 8))
            top.pack(fill='x', padx=10, pady=8)
            role_desc = {'A': 'A — donor · tensors to replace / add',
                         'B': 'B — base · layout + metadata (unselected kept)',
                         'C': 'C — output · new GGUF to write'}
            role_color = {'A': '#e879f9', 'B': '#60a5fa', 'C': '#34d399'}
            self.role_meta = {}
            for i, role in enumerate(('A', 'B', 'C')):
                ttk.Label(top, text=role_desc[role], foreground=role_color[role],
                          font=('Segoe UI', 9, 'bold'), width=44, anchor='w').grid(row=i, column=0, sticky='w', padx=6, pady=2)
                ttk.Entry(top, textvariable=self.role_path[role], width=72).grid(row=i, column=1, sticky='ew', padx=4, pady=2)
                ttk.Button(top, text='Browse…', command=lambda role=role: self._browse(role)).grid(row=i, column=2, padx=4, pady=2)
                self.role_meta[role] = ttk.Label(top, text='', style='Dim.TLabel', font=('Consolas', 8))
                self.role_meta[role].grid(row=i, column=3, sticky='e', padx=8, pady=2)
            bf = ttk.Frame(top)
            bf.grid(row=3, column=0, columnspan=4, sticky='ew', pady=(6, 0))
            ttk.Button(bf, text='Load A + B', style='Primary.TButton', command=self._load).pack(side='left', padx=4)
            ttk.Button(bf, text='Load B (extract only)', command=self._load_b).pack(side='left', padx=4)
            ttk.Button(bf, text='Extract selected → donor GGUF', command=self._extract).pack(side='right', padx=4)
            top.columnconfigure(1, weight=1)

            # ---- tensor card ----
            card = ttk.LabelFrame(r, text='  TENSORS   click a column header to sort  ', padding=(10, 6))
            card.pack(fill='both', expand=True, padx=10, pady=6)

            sm = ttk.Frame(card)
            sm.pack(fill='x', pady=(0, 2))
            self.sum_var = tk.StringVar(value='load A + B to preview C')
            self.sum_size_var = tk.StringVar(value='')
            self.sum_kv_var = tk.StringVar(value='')
            ttk.Label(sm, textvariable=self.sum_var, foreground='#aab2d0', font=('Segoe UI', 9, 'bold')).pack(side='left')
            ttk.Label(sm, textvariable=self.sum_size_var, foreground='#34d399', font=('Segoe UI', 9, 'bold')).pack(side='left', padx=16)
            ttk.Label(sm, textvariable=self.sum_kv_var, foreground='#e879f9', font=('Consolas', 8)).pack(side='right')

            ctl = ttk.Frame(card)
            ctl.pack(fill='x', pady=(0, 4))
            self.search_var = tk.StringVar()
            ttk.Entry(ctl, textvariable=self.search_var, width=26).pack(side='left', padx=(0, 6))
            self.search_var.trace_add('write', lambda *_: self._refresh_table())
            ttk.Button(ctl, text='Select all (in A)', command=lambda: self._quick_sel(None)).pack(side='left', padx=3)
            ttk.Button(ctl, text='Clear selection', command=self._clear_sel).pack(side='left', padx=3)
            self.cnt_var = tk.StringVar(value='0 selected')
            ttk.Label(ctl, textvariable=self.cnt_var, foreground='#34d399', font=('Segoe UI', 9, 'bold')).pack(side='right', padx=6)

            tf = ttk.Frame(card)
            tf.pack(fill='both', expand=True)
            self.tree = ttk.Treeview(tf, columns=self.COLS, show='headings', selectmode='extended')
            self.tree.tag_configure('sel',  background='#1d2f52', foreground='#cdd9f7')   # replaced from A
            self.tree.tag_configure('new',  background='#123a33', foreground='#99f6e4')   # donor-only, added
            self.tree.tag_configure('warn', background='#3a2f14', foreground='#fcd34d')   # kept from B (mismatch)
            self.tree.tag_configure('dim',  foreground='#5c6274')                        # donor-only, not selected
            self.tree.tag_configure('base', foreground='#aeb4c8')                        # plain B row
            for c in self.COLS:
                self.tree.column(c, width=self.COL_W[c], anchor='w' if c in ('name', 'a', 'b', 'c') else 'center')
                self.tree.heading(c, text=self.COL_L[c], command=lambda c=c: self._sort(c))
            vsb = ttk.Scrollbar(tf, orient='vertical', command=self.tree.yview)
            self.tree.configure(yscrollcommand=vsb.set)
            self.tree.pack(side='left', fill='both', expand=True)
            vsb.pack(side='right', fill='y')
            self.tree.bind('<Button-1>', self._on_click)
            ttk.Label(card,
                      text='✓ click to select · rows with "B —" exist only in A — select to append to C (e.g. blk.64 MTP) · ⚠ = shape mismatch, kept from B',
                      style='Dim.TLabel', font=('Segoe UI', 8)).pack(anchor='w', pady=(4, 0))

            # ---- action bar + log ----
            bot = ttk.Frame(r, style='Card.TFrame')
            bot.pack(fill='x', padx=10, pady=(0, 6))
            self.apply_btn = ttk.Button(bot, text='Apply → write C', style='Success.TButton', command=self._apply)
            self.apply_btn.pack(side='left', padx=4)
            self.cancel_btn = ttk.Button(bot, text='Cancel', state='disabled', command=self.cancel_ev.set)
            self.cancel_btn.pack(side='left', padx=4)
            self.prog = ttk.Progressbar(bot, length=340, style='Horizontal.TProgressbar', maximum=1000)
            self.prog.pack(side='left', padx=12, fill='x', expand=True)
            self.speed_var = tk.StringVar()
            ttk.Label(bot, textvariable=self.speed_var, foreground='#8b93b0', font=('Consolas', 9)).pack(side='right', padx=8)
            self.log = tk.Text(r, height=7, bg=self._field_bg, fg='#9ee493', font=('Consolas', 9),
                               insertbackground='white', state='disabled', relief='flat', highlightbackground='#262a37')
            self.log.pack(fill='x', padx=10, pady=(0, 8))

        def _log(self, msg: str):
            self.log.configure(state='normal')
            self.log.insert('end', msg + '\n')
            self.log.see('end')
            self.log.configure(state='disabled')

        # ---------- file roles ----------
        def _browse(self, role):
            if role == 'C':
                init = 'output.gguf'
                b = self.parsed.get('B')
                if b:
                    init = os.path.splitext(os.path.basename(b['path']))[0] + '_C.gguf'
                p = filedialog.asksaveasfilename(defaultextension='.gguf', filetypes=[('GGUF', '*.gguf')], initialfile=init)
            else:
                p = filedialog.askopenfilename(filetypes=[('GGUF', '*.gguf')])
            if p:
                self.role_path[role].set(p)

        def _parse_role(self, role):
            path = self.role_path[role].get().strip()
            if not path or not os.path.isfile(path):
                raise FileNotFoundError(f'{role}: file not found: {path!r}')
            if path in self.role_file and self.role_file[path] == role:
                return self.parsed[role]
            self._log(f'parsing {role}: {os.path.basename(path)} …')
            t0 = time.time()
            info = parse_gguf(path)
            self.parsed[role] = info
            self.role_file[path] = role
            self._set_meta(role, info)
            self._log(f'  {role}: {len(info["tensors"])} tensors, {len(info["kv"])} kv, '
                      f'{fmt(info["file_size"])}, data@{info["data_offset"]:,} ({time.time()-t0:.1f}s)')
            return info

        def _set_meta(self, role, info):
            self.role_meta[role].configure(
                text=f'{len(info["tensors"])} tensors · {len(info["kv"])} kv · {fmt(info["file_size"])}')

        def _load(self):
            try:
                a = self._parse_role('A')
                b = self._parse_role('B')
            except Exception as e:
                self._log(f'ERROR: {e}'); return
            rows = [TRow(t.name, a['by_name'].get(t.name), t) for t in b['tensors']]
            bnames = {t.name for t in b['tensors']}
            extra = [t for t in a['tensors'] if t.name not in bnames]
            rows += [TRow(t.name, t, None) for t in extra]  # A-only, in donor order
            self.rows = rows
            self.sel = set()  # no auto-selection — user picks manually
            self._refresh_table()
            self._log(f'loaded. B={len(b["tensors"])} tensors, A={len(a["tensors"])} tensors'
                      + (f', {len(extra)} A-only rows listed (not in B — select to add to C)' if extra else ''))

        def _load_b(self):
            """Load only B (no A needed) — enough for Extract selected."""
            try:
                b = self._parse_role('B')
            except Exception as e:
                self._log(f'ERROR: {e}'); return
            self.parsed.pop('A', None)
            self.role_file = {p: role for p, role in self.role_file.items() if role != 'A'}
            self.rows = [TRow(t.name, None, t) for t in b['tensors']]
            self.sel = set()
            self._refresh_table()
            self._log(f'loaded B only: {len(b["tensors"])} tensors — extract works; '
                      'for transfer (C preview/apply) use "Load A + B"')

        # ---------- selection ----------
        def _quick_sel(self, fam):
            names = {G.type_name(x) for x in fam} if fam else None
            for r in self.rows:
                if r.b is None or r.a is None:  # A-only rows: selected explicitly
                    continue
                if names is None or G.type_name(r.b.type) in names:
                    self.sel.add(r.name)
            self._refresh_table()

        def _clear_sel(self):
            self.sel.clear()
            self._refresh_table()

        def _on_click(self, ev):
            if self.tree.identify_column(ev.x) == '#1':
                iid = self.tree.identify_row(ev.y)
                if iid:
                    name = self.tree.item(iid)['values'][1]
                    self.sel.symmetric_difference_update((name,))
                    self._refresh_table()

        # ---------- table ----------
        def _c_res(self, r):
            """(source, spec, reason) — mirrors Plan.build exactly.
            source: 'A' | 'B' | None (dropped) · reason: 'ok' | 'kept' | 'new' | 'skip'"""
            sel = r.name in self.sel
            if r.b is not None:
                if sel:
                    if r.a is not None and r.a.dims == r.b.dims:
                        return 'A', r.a, 'ok'
                    return 'B', r.b, 'kept'
                return 'B', r.b, 'ok'
            if sel and r.a is not None:
                return 'A', r.a, 'new'
            return None, None, 'skip'

        def _row_values(self, r):
            a, b = r.a, r.b
            src, spec, reason = self._c_res(r)
            vals = ['✓' if r.name in self.sel else '☐',
                    r.name,
                    f'{a.type_name} · {fmt(a.size)}' if a else '—',
                    f'{b.type_name} · {fmt(b.size)}' if b else '—']
            if src is None:
                vals.append('— (not in C)')
            elif reason == 'new':
                vals.append(f'{spec.type_name} · {fmt(spec.size)}  (+)')
            elif reason == 'kept':
                vals.append(f'{spec.type_name} · {fmt(spec.size)}  ⚠ kept B')
            else:
                vals.append(f'{spec.type_name} · {fmt(spec.size)}  ({src})')
            before = b.size if b else 0
            after = spec.size if spec else 0
            vals.append(fmt_delta(after - before))
            return vals

        def _row_tags(self, r):
            src, spec, reason = self._c_res(r)
            if src is None:
                return ('dim',)
            if reason == 'kept':
                return ('warn',)
            if reason == 'new':
                return ('new',)
            if src == 'A':
                return ('sel',)
            return ('base',)

        def _sort(self, col):
            if self.sort_col == col:
                self.sort_dir *= -1
            else:
                self.sort_col, self.sort_dir = col, 1
            self._refresh_table()

        def _sort_key(self, r):
            col = self.sort_col
            a, b = r.a, r.b
            src, spec, reason = self._c_res(r)
            if col == 'sel':
                return (0 if r.name in self.sel else 1, r.name)
            if col == 'name':
                return r.name.lower()
            if col == 'a':
                return (1 if a is None else 0, a.size if a else 0)
            if col == 'b':
                return (1 if b is None else 0, b.size if b else 0)
            if col == 'c':
                return (1 if spec is None else 0, spec.size if spec else 0)
            before = b.size if b else 0
            return (spec.size if spec else 0) - before  # delta

        def _refresh_table(self):
            q = self.search_var.get().strip().lower()
            view = [r for r in self.rows if (not q) or (q in r.name.lower())]
            view.sort(key=lambda r: self._sort_key(r), reverse=self.sort_dir < 0)
            for iid in self.tree.get_children():
                self.tree.delete(iid)
            for r in view:
                self.tree.insert('', 'end', values=self._row_values(r), tags=self._row_tags(r))
            self._update_heading_marks()
            self.cnt_var.set(f'{len(self.sel)} selected')
            self._update_summary()

        def _update_heading_marks(self):
            for c in self.COLS:
                txt = self.COL_L[c]
                if c == self.sort_col:
                    txt += '  ▼' if self.sort_dir > 0 else '  ▲'
                self.tree.heading(c, text=txt)

        def _update_summary(self):
            a, b = self.parsed.get('A'), self.parsed.get('B')
            if not (a and b and self.rows):
                self.sum_var.set('load A + B to preview C')
                self.sum_size_var.set('')
                self.sum_kv_var.set('')
                return
            plan = Plan(b, a, self.sel)
            total = plan.header_len() + plan.total_data
            n_new = len(plan.extra)
            n_a = sum(1 for t, _, _ in plan.out_tensors if t == 'A') - n_new
            n_b = len(plan.out_tensors) - n_a - n_new
            d = total - b['file_size']
            self.sum_var.set(f'C: {len(plan.out_tensors)} tensors   from A: {n_a} · from B: {n_b} · added: {n_new}')
            self.sum_size_var.set(f'{fmt(b["file_size"])}  →  {fmt(total)}   (Δ {fmt_delta(d)})')
            self.sum_kv_var.set('   '.join(plan.kv_patches))

        # ---------- apply / extract ----------
        def _extract(self):
            if not self.parsed.get('B'):
                self._log('load B first (Load B or Load A + B)'); return
            b = self.parsed['B']
            sel = {n for n in self.sel if n in b['by_name']}
            skipped = len(self.sel) - len(sel)
            if skipped:
                self._log(f'note: {skipped} selected exist only in A — skipped (extract is from B)')
            if not sel:
                self._log('nothing selected (present in B)'); return
            p = filedialog.asksaveasfilename(defaultextension='.gguf', filetypes=[('GGUF', '*.gguf')],
                                              initialfile=f'donor-{len(sel)}-tensors.gguf')
            if not p:
                return
            try:
                n = extract_donor(b, sel, p, b['kv'].get('general.architecture', 'generic'))
                self._log(f'extracted {n} tensors → {p}')
            except Exception as e:
                self._log(f'extract ERROR: {e}')

        def _apply(self):
            if self.busy:
                return
            try:
                a = self._parse_role('A')
                b = self._parse_role('B')
            except Exception as e:
                self._log(f'ERROR: {e}'); return
            cpath = self.role_path['C'].get().strip()
            if not cpath:
                self._log('set output C path'); return
            if cpath.lower() in (a['path'].lower(), b['path'].lower()):
                self._log('output C must differ from A and B'); return
            if not self.sel:
                self._log('select at least one tensor'); return
            plan = Plan(b, a, self.sel)
            for w in plan.warnings:
                self._log('warn: ' + w)
            if plan.kv_patches:
                self._log('C metadata: ' + '   |   '.join(plan.kv_patches))
            self._log(f'plan: {len(plan.out_tensors)} tensors ({len(plan.extra)} added from A), '
                      f'{len(plan.spans)} spans, total ~{fmt(plan.header_len() + plan.total_data)}')
            self.busy = True
            self.cancel_ev.clear()
            self.apply_btn.configure(state='disabled')
            self.cancel_btn.configure(state='normal')
            self.prog['value'] = 0
            threading.Thread(target=self._apply_worker, args=(plan, cpath), daemon=True).start()

        def _apply_worker(self, plan, cpath):
            t0 = time.time()
            try:
                def cb(done, total, el):
                    self.q.put(('prog', done, total, el))
                write_output(plan, cpath, cb, self.cancel_ev.is_set)
                self.q.put(('done', cpath, plan, time.time() - t0))
            except InterruptedError:
                self.q.put(('cancel', None, None, None))
            except Exception as e:
                self.q.put(('err', str(e), None, None))

        def _poll(self):
            try:
                while True:
                    m = self.q.get_nowait()
                    if m[0] == 'prog':
                        _, done, total, el = m
                        self.prog['value'] = 1000 * done / max(total, 1)
                        sp = done / el if el > 0.05 else 0
                        self.speed_var.set(f'{done/1e9:.2f}/{total/1e9:.2f} GB  {sp/1e6:.0f} MB/s  {100*done/total:.1f}%')
                    elif m[0] == 'done':
                        _, cpath, plan, el = m
                        self.busy = False
                        self.apply_btn.configure(state='normal')
                        self.cancel_btn.configure(state='disabled')
                        self._log(f'wrote {cpath} in {el:.1f}s ({fmt(os.path.getsize(cpath))})')
                        self._verify_async(plan, cpath)
                    elif m[0] == 'cancel':
                        self.busy = False
                        self.apply_btn.configure(state='normal')
                        self.cancel_btn.configure(state='disabled')
                        self._log('cancelled, partial file removed')
                    elif m[0] == 'err':
                        self.busy = False
                        self.apply_btn.configure(state='normal')
                        self.cancel_btn.configure(state='disabled')
                        self._log(f'ERROR: {m[1]}')
                    elif m[0] == 'verify':
                        for line in m[1]:
                            self._log('verify: ' + line)
            except queue.Empty:
                pass
            self.root.after(100, self._poll)

        def _verify_async(self, plan, cpath):
            def w():
                res = verify_output(plan, cpath)
                self.q.put(('verify', res))
            threading.Thread(target=w, daemon=True).start()

    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == '__main__':
    _main_gui()
