# Ultra Doc-Intelligence

An AI-powered system for logistics document analysis. Upload a logistics document (Rate Confirmation, BOL, Shipment Instructions, Invoice, Pick List, etc.) and interact with it using natural language questions.

Runs **100% locally** — no cloud APIs, no internet required after setup. Works on any PC regardless of specs.

---

## Features

| Feature | Description |
|---------|-------------|
| **Document Upload** | PDF, DOCX, and TXT files |
| **RAG Q&A** | Ask natural language questions grounded strictly in document content |
| **Guardrails** | Multi-layer hallucination prevention |
| **Confidence Scoring** | Weighted score from retrieval similarity, chunk agreement, and answer coverage |
| **Structured Extraction** | Extract shipment data as clean JSON (shipment_id, shipper, consignee, rates, dates, line items) |
| **100% Local** | Fully offline — no cloud APIs required |
| **Universal Compatibility** | NVIDIA GPU, Apple Silicon, AMD, or CPU-only — auto-detected |

---

## Quick Start

> **Only two commands needed** after creating a conda environment.

```bash
# 1. Create a conda environment (one-time)
conda create -n ultra_doc python=3.11 -y
conda activate ultra_doc

# 2. Install all dependencies (auto-detects your hardware)
python install.py

# 3. Launch the UI
python api.py
```

Open **http://localhost:7860** in your browser. That's it.

On first run, the model downloads from HuggingFace Hub (~2.5–5 GB). Subsequent runs load from cache.

> **Important:** Gemma models are gated on HuggingFace. Before running, create a `.env` file in the project folder with your token:
> ```
> HF_TOKEN=hf_your_token_here
> ```
> Get your token at https://huggingface.co/settings/tokens and accept the model license at https://huggingface.co/google/gemma-4-E2B-it.

---

## Hardware Support

The installer auto-detects your hardware and selects the optimal configuration.

| Hardware | Backend | Memory Required | Speed |
|----------|---------|----------------|-------|
| **NVIDIA GPU (6 GB+ VRAM)** | NF4 4-bit (bitsandbytes) | ~3 GB VRAM | 25–35 tok/s |
| **NVIDIA GPU (3–6 GB VRAM)** | NF4 4-bit (bitsandbytes) | ~2 GB VRAM | 20–30 tok/s |
| **Apple Silicon (M1/M2/M3/M4)** | float16 (MPS) | ~4 GB unified mem | 15–25 tok/s |
| **CPU only (8 GB+ RAM)** | int8 weight-only (torchao) | ~5 GB RAM | 5–10 tok/s |
| **CPU only (4–8 GB RAM)** | int8, smaller model (1B) | ~3 GB RAM | 5–8 tok/s |

> **Minimum spec:** 4 GB RAM, any modern CPU (2015+). Under 4 GB RAM is not recommended.

---

## Auto Model Selection

The system automatically picks the right model size for your hardware:

| Condition | Model Selected |
|-----------|---------------|
| NVIDIA GPU with ≥ 4 GB VRAM | `google/gemma-4-E2B-it` (2B) |
| NVIDIA GPU with < 4 GB VRAM | `google/gemma-3-1b-it` (1B) |
| Apple Silicon (any) | `google/gemma-4-E2B-it` (2B) |
| CPU with ≥ 5 GB free RAM | `google/gemma-4-E2B-it` (2B) |
| CPU with < 5 GB free RAM | `google/gemma-3-1b-it` (1B) |

Override at any time:
```bash
# Use a specific model
set ULTRA_DOC_MODEL=google/gemma-3-1b-it    # Windows
export ULTRA_DOC_MODEL=google/gemma-3-1b-it  # Linux / Mac
```

---

## Supported File Formats

| Format | Extension | Status | How it works |
|--------|-----------|--------|--------------|
| **PDF** | `.pdf` | Supported | Text extracted directly from the PDF layer |
| **Word Document** | `.docx`, `.doc` | Supported | Text + tables extracted from the document structure |
| **Plain Text** | `.txt` | Supported | Loaded as-is |
| **Image (PNG/JPG/etc.)** | `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff` | Requires extra setup | OCR via Tesseract — needs separate binary install |

> **Images (PNG, JPG, etc.) need extra setup.**
> Install the Tesseract binary separately:
> - **Windows:** https://github.com/UB-Mannheim/tesseract/wiki
> - **Linux:** `sudo apt-get install tesseract-ocr`
> - **Mac:** `brew install tesseract`

> **Scanned PDFs are not supported.** If your PDF was created by scanning a paper document, the system returns empty content. Only text-based PDFs (where you can select text in a PDF viewer) work.

---

## Architecture

```
Document Upload
       |
  Document Processor  ─── PDF / DOCX / TXT / OCR
       |
   Chunker  ─── 800 char chunks, 100 overlap, table-aware
       |
  MiniLM Embedder  ─── 384-dim vectors (CPU, no GPU needed)
       |
   ChromaDB  ─── local persistent vector store
       |
  User Question ─► embed ─► top-3 similarity search
                                    |
                              RAG Engine
                                    |
                       LLM (Gemma, quantized)
                                    |
                    Guardrails + Confidence Scoring
                                    |
                               Answer + JSON
```

---

## Setup & Installation (Detailed)

### Prerequisites

- [Anaconda or Miniconda](https://www.anaconda.com/download) installed
- At least **4 GB RAM** (8 GB recommended)
- At least **5 GB free disk space** (model + packages)
- A HuggingFace account with your token set in `.env` (required — Gemma models are gated)

---

### Step 1 — Create a Conda environment

```bash
conda create -n ultra_doc python=3.11 -y
conda activate ultra_doc
```

---

### Step 2 — Configure HuggingFace token

Create a `.env` file in the project folder:

```
HF_TOKEN=hf_your_token_here
```

Get your token at https://huggingface.co/settings/tokens and accept the model license on HuggingFace.

---

### Step 3 — Install dependencies (automatic)

```bash
python install.py
```

The installer will:
- Detect your hardware (NVIDIA GPU / Apple Silicon / AMD / CPU-only)
- Measure your available RAM and VRAM
- Print a system summary with the recommended model tier
- Install the correct PyTorch wheel and all dependencies
- Verify every import works

Output example (NVIDIA GPU):
```
System Summary:
  RAM        : 16.0 GB total
  GPU        : NVIDIA detected  (CUDA 12.8)
  VRAM       : 6.0 GB
  Model tier : google/gemma-4-E2B-it
```

> **Manual alternative:**
> ```bash
> pip install -r requirements_gpu.txt   # NVIDIA GPU
> pip install -r requirements_cpu.txt   # CPU / Apple Silicon / AMD
> ```

---

### Step 4 — Run

```bash
python api.py
```

Open **http://localhost:7860** in your browser.

---

## Quantization Strategy

| Device | Method | VRAM / RAM | Library |
|--------|--------|-----------|---------|
| **NVIDIA GPU** | NF4 4-bit double quantization | ~3 GB VRAM | bitsandbytes |
| **Apple Silicon** | float16 (no quantization) | ~4 GB unified mem | native PyTorch MPS |
| **CPU** | int8 weight-only | ~5 GB RAM | torchao |

Auto-detects your device — no manual configuration needed.

If GPU runs out of memory during loading, the system automatically falls back to CPU.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FORCE_DEVICE` | auto-detect | `cpu` · `cuda` · `mps` |
| `ULTRA_DOC_MODEL` | auto-select | Any HuggingFace model ID |
| `HF_TOKEN` | — | HuggingFace token for gated model downloads (required) |

---

## CLI mode (optional)

```bash
python app.py <path_to_document>
```

---

## How to Use

> Sample documents are in the `Test files/` folder — pick list (DOCX), bill of lading (PDF), and skill test (DOCX).

1. **Upload a document** — PDF, DOCX, or TXT
2. Click **"Index Document"** — wait for: `Success: Indexed X chunks`
3. Type a question → click **"Get Answer"**
4. Click **"Extract Shipment JSON"** for structured field extraction

---

## Document Processing

### Chunking Strategy

- **Table-aware splitting** — table blocks kept as intact chunks
- **Recursive splitting** with separator hierarchy: `\n\n` → `\n` → `. ` → ` `
- **Chunk size:** 800 characters with 100-character overlap

### Retrieval

- **Embeddings:** `all-MiniLM-L6-v2` (384-dim, runs on CPU regardless of device)
- **Vector Store:** ChromaDB with cosine similarity
- **Top-K:** 3 most similar chunks retrieved per query
- **Similarity Threshold:** 0.15 minimum

---

## Guardrails

Three layers of hallucination prevention:

| Guardrail | Trigger | Action |
|-----------|---------|--------|
| **No Context** | No relevant chunks retrieved | Returns "Not found in document" |
| **Low Similarity** | Best chunk similarity < 0.15 | Refuses to answer |
| **Low Confidence** | Weighted confidence < 0.10 | Refuses to answer |

---

## Confidence Scoring

**Score = 0.4 × Retrieval Similarity + 0.3 × Chunk Agreement + 0.3 × Answer Coverage**

| Level | Score | Meaning |
|-------|-------|---------|
| High | ≥ 70% | Well-grounded answer |
| Medium | 40–70% | Partially grounded — review sources |
| Low | < 40% | May be blocked |

---

## Known Limitations

1. **Scanned PDFs** — OCR required; not enabled by default
2. **Very large documents** — Extraction truncates at configurable character limit (default 6,000 chars)
3. **Single document at a time** — No multi-document cross-query
4. **Non-English documents** — Prompts and embeddings are English-optimized
5. **Very low RAM (< 4 GB)** — May fail to load even the 1B model

---

## Project Structure

```
Ultra doc/
├── api.py                      # Gradio UI — run this!
├── app.py                      # CLI mode
├── install.py                  # Smart installer (auto-detects all hardware)
├── backend/
│   ├── config.py               # Central config (device, model tier, thresholds)
│   ├── document_processor.py   # Parse PDF/DOCX/TXT/images + chunking
│   ├── embedding_store.py      # Sentence-transformers + ChromaDB
│   ├── rag_engine.py           # RAG pipeline + LLM (CUDA/MPS/CPU)
│   ├── extractor.py            # Structured shipment data extraction
│   └── guardrails.py           # Guardrails + confidence scoring
├── Test files/                 # Sample documents for testing
├── requirements_gpu.txt        # NVIDIA GPU dependencies
├── requirements_cpu.txt        # CPU / Apple Silicon / AMD dependencies
├── .env.example                # Environment variable template
├── TEST_PLAN.md                # Fresh-environment test plan
└── README.md
```

---

## Tested On

| Component | Version |
|-----------|---------|
| **Python** | 3.11.15 |
| **PyTorch** | 2.11.0+cu128 |
| **Transformers** | 5.6.2 |
| **GPU** | NVIDIA GeForce RTX 4050 (6 GB VRAM) |
| **CUDA Driver** | 12.8 |
| **OS** | Windows 11 |

Last verified: April 2026 — clean `conda create` → `install.py` → `api.py` — all imports OK, model loads, Q&A and extraction working.

---

## License

This project is a POC / skill assessment submission.
