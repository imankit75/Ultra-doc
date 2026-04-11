# Ultra Doc-Intelligence

An AI-powered POC system for logistics document analysis. Upload a logistics document (Rate Confirmation, BOL, Shipment Instructions, Invoice, Pick List, etc.) and interact with it using natural language questions. Built as a simulated AI assistant inside a Transportation Management System (TMS).

Runs **100% locally** — no cloud APIs, no LM Studio, no internet required after setup.

---

## Features

| Feature | Description |
|---------|-------------|
| **Document Upload** | Supports PDF, DOCX, TXT, and image files (PNG, JPG via OCR) |
| **RAG Q&A** | Ask natural language questions grounded strictly in document content |
| **Guardrails** | Multi-layer hallucination prevention |
| **Confidence Scoring** | Weighted score from retrieval similarity, chunk agreement, and answer coverage |
| **Structured Extraction** | Extract shipment data as clean JSON (shipment_id, shipper, consignee, rates, dates, line items…) |
| **100% Local** | Runs fully offline with Gemma 3 4B (GGUF) — no cloud APIs required |
| **CPU + GPU** | Works on CPU-only machines; GPU gives a major speed boost |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      Gradio UI (api.py)                      │
│            Upload • Q&A • Structured Extraction              │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                      Backend Pipeline                        │
│                                                              │
│  ┌─────────────────┐   ┌──────────────┐   ┌──────────────┐  │
│  │ Document         │──▶│ Embedding    │──▶│ ChromaDB     │  │
│  │ Processor        │   │ (MiniLM-L6)  │   │ Vector Store │  │
│  │ (PDF/DOCX/TXT/   │   └──────────────┘   └──────┬───────┘  │
│  │  PNG/JPG + OCR)  │                              │          │
│  └─────────────────┘                               ▼          │
│                                                              │
│  ┌─────────────────┐   ┌──────────────┐   ┌──────────────┐  │
│  │ Guardrails &    │◀──│ RAG Engine   │◀──│ Retrieval    │  │
│  │ Confidence      │   │              │   │ (Top-K)      │  │
│  └────────┬────────┘   └──────┬───────┘   └──────────────┘  │
│           │                    │                              │
│           ▼                    ▼                              │
│  ┌──────────────────────────────────────┐                    │
│  │     Gemma 3 4B (GGUF, Local)        │                    │
│  │     via llama-cpp-python             │                    │
│  │     CPU (n_gpu_layers=0) or          │                    │
│  │     GPU (n_gpu_layers=-1)            │                    │
│  └──────────────────────────────────────┘                    │
└──────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Download the Model

Download `gemma-3-4b-it-Q4_K_M.gguf` from HuggingFace and place it at:

```
models/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_M.gguf
```

> **Download:** https://huggingface.co/bartowski/gemma-3-4b-it-GGUF

### 2A. Run with Docker (Recommended)

#### CPU (works on any machine)

```bash
docker compose up --build
```

#### GPU (NVIDIA — requires `nvidia-container-toolkit`)

```bash
docker compose -f docker-compose.gpu.yml up --build
```

Open **http://localhost:7860** in your browser.

---

### 2B. Run Locally (without Docker)

#### Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** For GPU support locally, install llama-cpp-python with CUDA:
> ```bash
> CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
> ```

#### Set GPU mode (optional)

```bash
# CPU (default)
export N_GPU_LAYERS=0

# Full GPU offload
export N_GPU_LAYERS=-1
```

#### Run the UI

```bash
python api.py
```

#### Run CLI mode

```bash
python app.py <path_to_document>
```

---

## Docker Reference

| File | Purpose |
|------|---------|
| `Dockerfile` | CPU build — runs on any machine |
| `Dockerfile.gpu` | GPU build — NVIDIA CUDA 12.1 |
| `docker-compose.yml` | CPU compose (default) |
| `docker-compose.gpu.yml` | GPU compose |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `N_GPU_LAYERS` | `0` | `0` = CPU only · `-1` = all layers on GPU |

### Mounted Volumes

| Mount | Purpose |
|-------|---------|
| `./models:/app/models` | GGUF model files (read-only) |
| `./uploads:/app/uploads` | Uploaded documents (persistent) |
| `./chroma_db:/app/chroma_db` | Vector store (persistent) |

---

## Document Processing

### Chunking Strategy

- **Table-aware splitting** — preserves `TABLE START/END` blocks as intact chunks
- **Recursive splitting** with separator hierarchy: `\n\n` → `\n` → `. ` → ` `
- **Chunk size:** 800 characters
- **Overlap:** 100 characters — ensures context continuity at boundaries
- Sentence-boundary aware: avoids splitting mid-sentence when possible

### Why this approach?
Logistics documents contain structured data (addresses, dates, rates) mixed with free text and tables. Table-aware splitting keeps row data together while recursive splitting preserves logical blocks (paragraphs → sentences).

---

## Retrieval Method

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, runs on CPU)
- **Vector Store:** ChromaDB with cosine similarity
- **Top-K:** 5 most similar chunks retrieved per query
- **Similarity Threshold:** 0.15 minimum (below → refuse to answer)

### Why MiniLM?
Fast, lightweight (~22M params), runs without GPU, performs well on short-to-medium text — ideal for logistics document chunks.

---

## Guardrails Approach

Three layers of hallucination prevention:

| Guardrail | Trigger | Action |
|-----------|---------|--------|
| **No Context** | No relevant chunks retrieved | Returns "Not found in document" |
| **Low Similarity** | Best chunk similarity < 0.15 | Refuses to answer |
| **Low Confidence** | Weighted confidence < 0.10 | Refuses to answer |

The LLM system prompt strictly instructs the model to:
- **Only** answer from the provided context
- Say "Not found in the provided document" when information is missing
- Quote supporting text as evidence
- Handle tabular data with `[HEADER]` and `[ROW]` markers

---

## Confidence Scoring Method

Multi-factor weighted confidence score:

| Factor | Weight | Description |
|--------|--------|-------------|
| **Retrieval Similarity** | 40% | Average cosine similarity of top-K retrieved chunks |
| **Chunk Agreement** | 30% | Fraction of retrieved chunks above similarity threshold |
| **Answer Coverage** | 30% | Fraction of answer's meaningful words found in source chunks |

**Final Score = 0.4 × Similarity + 0.3 × Agreement + 0.3 × Coverage**

Color coded:
- 🟢 **High** (≥ 70%) — Confident, well-grounded answer
- 🟡 **Medium** (40–70%) — Partially grounded, review sources
- 🔴 **Low** (< 40%) — Low confidence, answer may be blocked

---

## Known Failure Cases

1. **Scanned PDFs / Image-only PDFs** — OCR requires Tesseract; quality depends on image clarity
2. **Very large documents** — Extraction truncates at 12K characters to fit LLM context
3. **Multi-document queries** — System processes one document at a time
4. **Ambiguous field names** — Structured extraction may miss non-standard field labels
5. **Non-English documents** — Embeddings and prompts are English-optimized

---

## Improvement Ideas

- **Hybrid search** — Combine vector similarity with BM25 keyword search
- **Multi-document support** — Query across multiple indexed documents
- **Streaming responses** — Stream LLM output token-by-token for better UX
- **Fine-tuned extraction** — Train a specialized model on logistics schemas
- **Evaluation framework** — Automated test suite with ground-truth Q&A pairs
- **Agentic workflow** — Multi-step reasoning for complex multi-hop questions

---

## Project Structure

```
Ultra doc/
├── api.py                      # Gradio UI — run this!
├── app.py                      # CLI mode (interactive terminal Q&A)
├── backend/
│   ├── __init__.py
│   ├── config.py               # Central config (reads N_GPU_LAYERS from env)
│   ├── document_processor.py   # Parse PDF/DOCX/TXT/images + table-aware chunking
│   ├── embedding_store.py      # Sentence-transformers + ChromaDB
│   ├── rag_engine.py           # RAG pipeline + Gemma 3 4B LLM calls
│   ├── extractor.py            # Structured shipment data extraction
│   └── guardrails.py           # Hallucination guardrails + confidence scoring
├── tessdata/                   # Tesseract OCR language data
│   └── eng.traineddata
├── models/                     # GGUF model files (NOT committed — too large)
│   └── .gitkeep                # Placeholder with download instructions
├── Dockerfile                  # CPU Docker build
├── Dockerfile.gpu              # GPU Docker build (NVIDIA CUDA 12.1)
├── docker-compose.yml          # CPU compose
├── docker-compose.gpu.yml      # GPU compose
├── requirements.txt
├── .env.example
├── .gitignore
├── .dockerignore
└── README.md
```

---

## UI Actions

| Action | Description |
|--------|-------------|
| **Index Document** | Upload and process a document (PDF, DOCX, TXT, images) |
| **Get Answer** | Ask a natural language question about the indexed document |
| **Extract Shipment JSON** | Extract all structured data fields as clean JSON |

---

## License

This project is a POC / skill assessment submission.
