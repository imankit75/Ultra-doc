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

The system follows a straightforward pipeline:

1. **Gradio UI** — The user uploads a document and asks questions through a simple web interface.
2. **Document Processor** — Parses the file (PDF, DOCX, or TXT) and splits it into smart chunks, keeping tables intact.
3. **Embeddings** — Each chunk is converted into a vector using the MiniLM-L6 sentence transformer model running locally.
4. **ChromaDB** — The vectors are stored in a local ChromaDB database for fast similarity search.
5. **Retrieval** — When a question is asked, the top 5 most relevant chunks are retrieved from ChromaDB.
6. **RAG Engine** — The retrieved chunks are assembled into a context prompt and sent to the LLM.
7. **Gemma 3 4B (local)** — The fully local GGUF model generates an answer grounded strictly in the retrieved context.
8. **Guardrails + Confidence** — The answer is checked for reliability before being returned to the user.

---

## Setup & Installation

### Prerequisites

Before starting, make sure you have:

- [Anaconda or Miniconda](https://www.anaconda.com/download) installed
- At least **8 GB of free RAM**
- At least **5 GB of free disk space** (for the model + packages)

---

### Step 1 — Create a Conda environment

```bash
conda create -n ultra_doc python=3.11 -y
conda activate ultra_doc
```

---

### Step 2 — Install llama-cpp-python (via conda-forge)

> **Why conda and not pip?**
> `llama-cpp-python` compiles a C++ library during install. On Windows, pip requires Visual Studio Build Tools (which most people don't have). The conda-forge version ships a pre-built binary — no compiler needed.

```bash
conda install -c conda-forge llama-cpp-python -y
```

This may take a few minutes to resolve and download.

---

### Step 3 — Install all dependencies

This installs everything including PyTorch (CPU). The `requirements.txt` file handles the correct CPU-only torch wheel automatically — no extra commands needed.

```bash
pip install -r requirements.txt
```

---

### Step 4 — Download the model

Download the `gemma-3-4b-it-Q4_K_M.gguf` file and place it **exactly** at:

```
models/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_M.gguf
```

> **Download link:** https://huggingface.co/bartowski/gemma-3-4b-it-GGUF
>
> Look for the file named `gemma-3-4b-it-Q4_K_M.gguf` (~2.5 GB).

---

### Step 5 — Run the UI

```bash
python api.py
```

Open **http://localhost:7860** in your browser. The app is ready.

---

### GPU Acceleration (optional)

If you have an NVIDIA GPU, you can run inference significantly faster. After completing the steps above, replace the llama-cpp install with the CUDA version:

```bash
conda install -c conda-forge llama-cpp-python=*=*cuda* -y
```

Then tell the app to use the GPU:

```bash
# Windows
set N_GPU_LAYERS=-1

# Linux / Mac
export N_GPU_LAYERS=-1
```

---

### CLI mode (optional)

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
