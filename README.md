# Ultra Doc-Intelligence

An AI-powered POC system for logistics document analysis. Upload a logistics document (Rate Confirmation, BOL, Shipment Instructions, Invoice, Pick List, etc.) and interact with it using natural language questions. Built as a simulated AI assistant inside a Transportation Management System (TMS).

Runs **100% locally** — no cloud APIs, no internet required after setup.

---

## Features

| Feature | Description |
|---------|-------------|
| **Document Upload** | PDF, DOCX, and TXT files |
| **RAG Q&A** | Ask natural language questions grounded strictly in document content |
| **Guardrails** | Multi-layer hallucination prevention |
| **Confidence Scoring** | Weighted score from retrieval similarity, chunk agreement, and answer coverage |
| **Structured Extraction** | Extract shipment data as clean JSON (shipment_id, shipper, consignee, rates, dates, line items…) |
| **100% Local** | Fully offline with Gemma 3 4B (GGUF) — no cloud APIs required |

---

## Supported File Formats

| Format | Extension | Status | How it works |
|--------|-----------|--------|--------------|
| **PDF** | `.pdf` | ✅ Supported | Text extracted directly from the PDF layer |
| **Word Document** | `.docx`, `.doc` | ✅ Supported | Text + tables extracted from the document structure |
| **Plain Text** | `.txt` | ✅ Supported | Loaded as-is |
| **Image (PNG/JPG/etc.)** | `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff` | ⚠️ Requires extra setup | OCR via Tesseract — needs separate binary install |

> [!CAUTION]
> **Images (PNG, JPG, etc.) do NOT work out of the box.**
> The code supports it in principle, but requires the **Tesseract binary** installed separately on your system:
> - **Windows:** https://github.com/UB-Mannheim/tesseract/wiki
> - **Linux:** `sudo apt-get install tesseract-ocr`
> - **Mac:** `brew install tesseract`
>
> Without the Tesseract binary, uploading an image will throw an error.

> [!CAUTION]
> **Images embedded inside a PDF or DOCX are NOT supported — at all.**
> If your PDF was scanned (pages are images, not selectable text), the system will return empty content — no error, just no data.
>
> ✅ **Works:** Text-based PDF (you can select/copy text in a PDF viewer)
> ✅ **Works:** Normal DOCX with typed text and tables
> ❌ **Does NOT work:** Scanned PDF (each page is a photo)
> ❌ **Does NOT work:** DOCX with embedded screenshots or image-only pages
> ❌ **Does NOT work:** PNG/JPG without Tesseract binary installed


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
│  │ (PDF/DOCX/TXT)    │   └──────────────┘   └──────┬───────┘  │
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
│  └──────────────────────────────────────┘                    │
└──────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Create and activate a Conda environment

```bash
conda create -n ultra_doc python=3.11 -y
conda activate ultra_doc
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU acceleration (optional):** Install llama-cpp-python with CUDA support for faster inference:
> ```bash
> CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall
> ```
> Then set the environment variable: `set N_GPU_LAYERS=-1` (Windows) or `export N_GPU_LAYERS=-1` (Linux/Mac)

### 3. Download the model

Download `gemma-3-4b-it-Q4_K_M.gguf` and place it at:
```
models/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_M.gguf
```

> **Download:** https://huggingface.co/bartowski/gemma-3-4b-it-GGUF

### 4. Run the Gradio UI

```bash
python api.py
```

Open **http://localhost:7860** in your browser.

### 5. (Optional) CLI mode

```bash
python app.py <path_to_document>
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `N_GPU_LAYERS` | `0` | `0` = CPU only · `-1` = all layers on GPU |

---

## How to Use

> **Sample documents to test with are available in the `Test files/` folder** — includes a pick list (DOCX) and a bill of lading (PDF).

1. **Upload a document** — PDF, DOCX, TXT, or standalone image (PNG/JPG)
2. Click **"Index Document"** — wait for the status message: `Success: Indexed X chunks`
3. Type a question in the question box → click **"Get Answer"**
4. Click **"Extract Shipment JSON"** to extract all structured fields as clean JSON

---

## Document Processing

### Chunking Strategy

- **Table-aware splitting** — preserves table blocks as intact chunks
- **Recursive splitting** with separator hierarchy: `\n\n` → `\n` → `. ` → ` `
- **Chunk size:** 800 characters
- **Overlap:** 100 characters — ensures context continuity at boundaries
- Sentence-boundary aware: avoids splitting mid-sentence when possible

### Why this approach?
Logistics documents contain structured data (addresses, dates, rates) mixed with free text and tables. Table-aware splitting keeps row data together while recursive splitting preserves logical blocks.

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

1. **Scanned PDFs / Image-only PDFs** — OCR quality depends on image clarity
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
├── tessdata/
│   └── eng.traineddata         # Tesseract OCR language data
├── models/                     # Place GGUF model here (not committed to git)
│   └── gemma-3-4b-it-GGUF/
│       └── gemma-3-4b-it-Q4_K_M.gguf
├── requirements.txt
├── .env.example
└── README.md
```

---

## License

This project is a POC / skill assessment submission.
