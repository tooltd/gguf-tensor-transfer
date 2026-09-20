# GGUF Tensor Transfer

Stream any tensors you pick from one GGUF file (**A — donor**) into another
(**B — base**) to create a new file (**C**). Selected tensors are copied
byte-exact from A; everything else is streamed from B — no re-quantization,
no full-file RAM load.

![GGUF Tensor Transfer UI](screenshots/ui.png)

## Features

- **Per-tensor selection** — the table shows every tensor with its quant
  type and size in A, B, and the resulting C. Click ✓ to select, click
  column headers to sort, search box filters by name.
- **Append donor-only tensors** — tensors that exist only in A (e.g. an
  extra transformer block like the MTP `blk.64` of Qwen3.5/3.6) are listed
  at the bottom and can be added to C; the matching metadata KVs
  (`block_count`, `nextn_predict_layers`) are synced automatically.
- **Shape-safe** — a tensor is taken from A only when its dimensions match;
  otherwise B's copy is kept and a ⚠ warning is shown.
- **Extract → donor GGUF** — load just B (**Load B (extract only)**), select
  tensors, and export them to a small standalone donor GGUF.
- **Safe write** — progress bar with live MB/s, cancel removes the partial
  file, and a quick verification runs after the write.

## Requirements

- Python 3.10+
- `pip install -r requirements.txt`

## Usage

```
python gguf_transfer_gui.py
```

1. Pick **A** (donor) and **B** (base) → **Load A + B**. Nothing is
   pre-selected — choose your tensors manually.
2. Set the **C** output path → **Apply → write C**.

Files stream in 256 MB chunks through a two-thread pipeline, so a ~12 GB
transfer peaks at only a few hundred MB of RAM.

## Files

| File | Role |
|------|------|
| `gguf_transfer_gui.py` | GUI + plan/write/verify core |
| `_gguf_fast.py` | Minimal streaming GGUF header parser/builder (stdlib only) |
