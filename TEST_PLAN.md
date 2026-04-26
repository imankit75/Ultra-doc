# Fresh-Environment Test Plan

End-to-end verification that Ultra Doc-Intelligence installs and runs from scratch on a clean machine.

---

## Prerequisites

- `conda` installed and on PATH
- NVIDIA GPU + driver (for GPU path) — `nvidia-smi` should work
- `.env` file at project root contains `HF_TOKEN=...` (Gemma 4 is gated)
- Project working directory: `C:\Users\ankit\Machine Learning\Agentic AI\Ultra doc`

---

## Phase 1 — Create fresh conda env

```powershell
conda create -n ultra_doc_fresh python=3.11 -y
```

**Verify:**
```powershell
conda env list
# Expect to see: ultra_doc_fresh
```

---

## Phase 2 — Install dependencies via install.py

Get the env's Python path:
```powershell
$ENV_PY = "C:\Users\ankit\anaconda3\envs\ultra_doc_fresh\python.exe"
```

Run installer:
```powershell
& $ENV_PY install.py
```

**Expected output:**
- `Platform : Windows AMD64 | Python 3.11.x`
- `NVIDIA GPU: detected`
- `CUDA driver: 13.0` (or whatever the host has)
- `Backend  : GPU  — bitsandbytes NF4 4-bit quantization`
- `Torch    : CUDA wheel (cu128) from ...`
- Pip installs torch + everything in `requirements_gpu.txt`
- `Done! All dependencies installed.`

**Verify imports:**
```powershell
& $ENV_PY -c "import torch; print('cuda:', torch.cuda.is_available())"
& $ENV_PY -c "import transformers; print('transformers:', transformers.__version__)"
& $ENV_PY -c "import torchao; print('torchao OK')"
& $ENV_PY -c "import bitsandbytes; print('bnb OK')"
& $ENV_PY -c "import gradio, chromadb, sentence_transformers, fitz, docx; print('all OK')"
```

Expectations:
- `cuda: True`
- `transformers: 5.x.x`
- All imports OK without errors

**If install fails:** capture the failing line, fix `requirements_gpu.txt` or `install.py`, recreate the env.

---

## Phase 3 — Verify HF token loads from .env

```powershell
& $ENV_PY -c "from dotenv import load_dotenv; load_dotenv(); import os; print('HF_TOKEN set:', bool(os.getenv('HF_TOKEN')))"
```

Expect: `HF_TOKEN set: True`

---

## Phase 4 — Launch the UI (model load smoke test)

```powershell
& $ENV_PY api.py
```

**Expected terminal output (in order):**
```
Loading model — this may take a moment on first run (downloads ~2-3 GB)...
============================================================
  MODEL LOADING
============================================================
  Source  : HuggingFace Hub
  Model   : google/gemma-4-E2B-it
  Device  : CUDA
  HF token: SET
============================================================
[1/4] Loading tokenizer from HF Hub...
      Tokenizer loaded.

[2/3] Loading model with bitsandbytes NF4 4-bit quantization...
      ...
      Model loaded + quantized in <N>s.

[3/3] Verifying GPU placement...
      GPU   : NVIDIA GeForce RTX 4050 ...
      VRAM  : ~3.0 GB used / 6.0 GB total  (~3.0 GB free)

  ✓ Model ready on CUDA!
============================================================

Model ready.
Running on local URL:  http://127.0.0.1:7860
Wi-Fi Access:          http://<wifi_ip>:7860
```

**If model load fails:** check the error against the troubleshooting matrix in this doc (below). Common: `401 gated repo` (token missing), `device-side assert` (VRAM overflow), `Params4bit error` (transformers/bnb mismatch).

---

## Phase 5 — UI functional test (manual, in browser)

Open http://127.0.0.1:7860.

### 5a — Upload + index
1. Upload a sample PDF/DOCX from `Test docs/`
2. Click **Index Document**
3. Verify status: `Success: Indexed <filename>! (N chunks)`

**Expected terminal logs:**
```
[DOC] Parsing .PDF file: <name>
[DOC] Parsed N page(s). Chunking text...
[DOC] Created M chunk(s) from N page(s).
[EMBED] Embedding M chunk(s)...
[EMBED] Embedding done in <t>s. Storing in ChromaDB...
[EMBED] Indexed M chunk(s) successfully.
```

### 5b — Question answering
1. Type a question in "Ask a Question"
2. Click **Get Answer**
3. Verify: response appears with confidence label

**Expected terminal logs:**
```
[RAG] Question received (Q&A): <question>
[RAG] Retrieving chunks from vector store...
[RAG] Retrieved N chunk(s).
      Top similarity: 0.xxx
[LLM] Inference starting...
      Prompt tokens : ...
      Output tokens : ...
      Time          : <t>s  (xx.x tok/s)
[RAG] Confidence: xx.x% | Guardrail: PASS
```

### 5c — Structured extraction
1. Click **Extract Shipment JSON**
2. Verify: JSON appears with document_type, line_items, etc.

**Expected terminal logs:**
```
[EXTRACT] Starting structured extraction: <filename>
[EXTRACT] Reading full document text...
[EXTRACT] Document text: N,NNN chars
[EXTRACT] Sending to LLM (max 1024 tokens output)...
[LLM] Inference starting...
      ...
[EXTRACT] Done. Document type: ... | Line items: N
```

---

## Phase 6 — Cleanup (only after PASS confirmed)

```powershell
# Stop the UI: Ctrl+C in the terminal
conda deactivate
conda env remove -n ultra_doc_fresh -y
```

---

## Troubleshooting matrix

| Symptom | Likely cause | Fix |
|---|---|---|
| `401 gated repo` | HF_TOKEN missing or invalid | Check `.env`, verify token at huggingface.co/settings/tokens, accept Gemma 4 license |
| `device-side assert triggered` | VRAM overflow | Confirm bnb NF4 is being used, not float16 |
| `Params4bit.__new__() got unexpected kwarg` | transformers/bitsandbytes API mismatch | Pin `transformers>=5.0.0` and latest bitsandbytes |
| `Cannot combine Quantization and loading from GGUF` | code branch hit local GGUF path | Confirm `HF_TOKEN` is set so use_local=False |
| `ImportError: Requires mslk` | Tried `Int4WeightOnlyConfig` | We use `Int8WeightOnlyConfig` on CPU now |
| `+cpu wheel not upgrading to cuXXX` | pip skips upgrade when version satisfies | install.py uninstalls torch first — verify it ran |
| Port 7860 in use | Previous Python process still running | `Get-Process python | Stop-Process -Force` |

---

## Pass criteria

All of the following must be true:
- [ ] `python install.py` completes without errors
- [ ] All required imports succeed
- [ ] `api.py` reaches "Model ready" without exceptions
- [ ] UI loads at http://127.0.0.1:7860
- [ ] Document upload + index produces correct logs
- [ ] Q&A produces an answer + sensible confidence
- [ ] Extract Shipment JSON produces a valid dict (not all-null)
- [ ] No tracebacks in terminal during any operation

If all checked → tear down env (Phase 6) and tag the commit as a known-good baseline.
