"""Lightweight GGUF parser/builder (gguf-py format, struct-only, no numpy).

String format: [u64 len][bytes]  (gguf-py convention).
KV value types: UINT8=0 INT8=1 UINT16=2 INT16=3 UINT32=4 INT32=5 FLOAT32=6
                BOOL=7 STRING=8 ARRAY=9 UINT64=10 INT64=11 FLOAT64=12
Tensor entry:  name-str, u32 ndim, u64 dims[ndim] (as stored in file), u32 type, u64 offset
               (packed tight, no per-entry padding; header padded to alignment at end)
Note: dims are handled as STORED order everywhere (parse returns stored, pack writes as-is).
      gguf-py readers apply reversed() to obtain the logical shape.
"""
from __future__ import annotations
import struct

MAGIC = 0x46554747  # 'GGUF'

V_U8, V_I8, V_U16, V_I16, V_U32, V_I32, V_F32, V_BOOL, V_STR, V_ARR, V_U64, V_I64, V_F64 = range(13)
_SCALAR_FMT = {
    V_U8: 'B', V_I8: 'b', V_U16: '<H', V_I16: '<h', V_U32: '<I', V_I32: '<i',
    V_F32: '<f', V_BOOL: '?', V_U64: '<Q', V_I64: '<q', V_F64: '<d',
}
_SCALAR_SIZE = {V_U8: 1, V_I8: 1, V_U16: 2, V_I16: 2, V_U32: 4, V_I32: 4,
                V_F32: 4, V_BOOL: 1, V_U64: 8, V_I64: 8, V_F64: 8}


class KV:
    __slots__ = ('key', 'vtype', 'value')
    def __init__(self, key: str, vtype: int, value):
        self.key, self.vtype, self.value = key, vtype, value
    def __repr__(self):
        v = self.value
        if isinstance(v, str): v = v[:40]
        if isinstance(v, list): v = f'list[{len(v)}]'
        return f'KV({self.key!r}, {self.vtype}, {v!r})'


class TensorInfo:
    __slots__ = ('name', 'ndim', 'dims', 'type', 'offset')
    def __init__(self, name, ndim, dims, t, off):
        self.name, self.ndim, self.dims, self.type, self.offset = name, ndim, dims, t, off
    @property
    def n_elements(self):
        n = 1
        for d in self.dims: n *= d
        return n
    def __repr__(self):
        return f'TensorInfo({self.name!r}, {self.dims}, t={self.type}, off={self.offset:,})'


class Header:
    def __init__(self):
        self.version = 0
        self.kv: list[KV] = []
        self.tensors: list[TensorInfo] = []
        self.alignment = 32  # GGUF default when general.alignment KV is absent
        self.kv_raw: bytes = b''          # raw KV bytes (start..end of KV section)
        self.ti_end = 0                    # offset where tensor-info section ends (before padding)
        self.data_offset = 0


def _read_value(f, o, vtype):
    fmt = _SCALAR_FMT.get(vtype)
    if fmt is not None:
        size = _SCALAR_SIZE[vtype]
        raw = f.read(size)
        v = struct.unpack(fmt, raw)[0]
        return o + size, v
    if vtype == V_STR:
        l = struct.unpack('<Q', f.read(8))[0]
        s = f.read(l).decode('utf-8')
        return o + 8 + l, s
    if vtype == V_ARR:
        itype = struct.unpack('<I', f.read(4))[0]
        alen = struct.unpack('<Q', f.read(8))[0]
        o += 12
        vals = []
        for _ in range(alen):
            o, v = _read_value(f, o, itype)
            vals.append(v)
        return o, vals
    raise ValueError(f'unknown vtype {vtype}')


def parse_header(f) -> Header:
    """Parse a GGUF header from file object f (left at end of header)."""
    f.seek(0)
    magic, version = struct.unpack('<II', f.read(8))
    if magic != MAGIC:
        raise ValueError(f'bad magic {magic:#x}')
    tensor_count, kv_count = struct.unpack('<QQ', f.read(16))
    h = Header()
    h.version = version

    kv_start = f.tell()
    for _ in range(kv_count):
        keylen = struct.unpack('<Q', f.read(8))[0]
        key = f.read(keylen).decode('utf-8')
        vtype = struct.unpack('<I', f.read(4))[0]
        _, value = _read_value(f, f.tell(), vtype)
        h.kv.append(KV(key, vtype, value))
    kv_end = f.tell()
    h.kv_raw_range = (kv_start, kv_end)

    for _ in range(tensor_count):
        keylen = struct.unpack('<Q', f.read(8))[0]
        name = f.read(keylen).decode('utf-8')
        ndim = struct.unpack('<I', f.read(4))[0]
        dims = list(struct.unpack(f'<{ndim}Q', f.read(8 * ndim)))
        t = struct.unpack('<I', f.read(4))[0]
        off = struct.unpack('<Q', f.read(8))[0]
        h.tensors.append(TensorInfo(name, ndim, dims, t, off))
    h.ti_end = f.tell()

    align = 32  # GGUF spec default (gguf-py GGUF_DEFAULT_ALIGNMENT)
    for kv in h.kv:
        if kv.key == 'general.alignment' and kv.vtype == V_U32:
            align = kv.value
    h.alignment = align
    h.data_offset = h.ti_end + (-h.ti_end % align) if h.ti_end % align else h.ti_end
    return h


def _pack_str(s: str) -> bytes:
    b = s.encode('utf-8')
    return struct.pack('<Q', len(b)) + b


def pack_value(vtype: int, value) -> bytes:
    fmt = _SCALAR_FMT.get(vtype)
    if fmt is not None:
        return struct.pack(fmt, value)
    if vtype == V_STR:
        return _pack_str(value)
    if vtype == V_ARR:
        # value: (itype, [items])
        itype, items = value
        out = struct.pack('<I', itype) + struct.pack('<Q', len(items))
        for it in items:
            out += pack_value(itype, it)
        return out
    raise ValueError(f'unknown vtype {vtype}')


def pack_kv(kv: KV) -> bytes:
    return _pack_str(kv.key) + struct.pack('<I', kv.vtype) + pack_value(kv.vtype, kv.value)


def pack_tensor_entry(t: TensorInfo) -> bytes:
    b = _pack_str(t.name)
    b += struct.pack('<I', t.ndim)
    for d in t.dims:  # dims are stored order, write as-is
        b += struct.pack('<Q', d)
    b += struct.pack('<I', t.type)
    b += struct.pack('<Q', t.offset)
    return b


def build_header_bytes(kv: list[KV], tensors: list[TensorInfo], alignment: int = 32) -> bytes:
    out = struct.pack('<IIQQ', MAGIC, 3, len(tensors), len(kv))
    for k in kv:
        out += pack_kv(k)
    for t in tensors:
        out += pack_tensor_entry(t)
    pad = (-len(out)) % alignment
    return out + b'\0' * pad


def tensor_bytes(ttype: int, n_elements: int) -> int:
    """Byte size of a tensor given (module) type id and element count. Uses gguf constants."""
    from gguf.constants import GGML_QUANT_SIZES
    elems, ts = GGML_QUANT_SIZES[ttype]
    assert n_elements % elems == 0
    return n_elements * ts // elems


def type_name(ttype: int) -> str:
    from gguf.constants import GGMLQuantizationType
    try:
        return GGMLQuantizationType(ttype).name
    except ValueError:
        return f'type{ttype}'
