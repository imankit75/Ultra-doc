"""
Ultra Doc-Intelligence — CLI (No UI)
A command-line tool for logistics document Q&A, RAG, and structured extraction.
Uses Gemma 3 4B (local GGUF) — fully offline, no LM Studio, no cloud APIs.
"""

import os

# Fix OpenMP DLL conflict between llama-cpp-python and PyTorch on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import uuid
import shutil

from backend.config import config
from backend.document_processor import process_document
from backend.embedding_store import embedding_store
from backend.rag_engine import ask_question
from backend.extractor import extract_shipment_data


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def print_banner():
    print()
    print("=" * 60)
    print("  🔍 Ultra Doc-Intelligence — CLI Mode")
    print("  Model: Gemma 3 4B (local GGUF)")
    print("  100% offline • No LM Studio • No cloud APIs")
    print("=" * 60)
    print()


def print_separator():
    print("-" * 60)


def upload_and_process(file_path: str) -> tuple[str, str]:
    """Process and index a document. Returns (doc_id, save_path)."""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

    os.makedirs(config.upload_dir, exist_ok=True)
    doc_id = str(uuid.uuid4())[:8]
    filename = os.path.basename(file_path)
    save_path = os.path.join(config.upload_dir, f"{doc_id}_{filename}")
    shutil.copy2(file_path, save_path)

    print(f"📄 Processing: {filename}")
    print(f"🆔 Doc ID: {doc_id}")
    print()

    # Parse & chunk
    print("⏳ Parsing document...")
    try:
        chunks = process_document(save_path)
    except Exception as e:
        print(f"❌ Parse error: {e}")
        sys.exit(1)

    # Embed & index
    print("⏳ Creating embeddings & indexing...")
    try:
        num_chunks = embedding_store.add_document(doc_id, chunks)
    except Exception as e:
        print(f"❌ Indexing error: {e}")
        sys.exit(1)

    print(f"✅ Indexed! {num_chunks} chunks created.")
    print()
    return doc_id, save_path


def run_qa(doc_id: str, question: str):
    """Ask a question about the indexed document."""
    print_separator()
    print(f"❓ Question: {question}")
    print()

    result = ask_question(doc_id=doc_id, question=question)

    # Answer
    answer = result["answer"]
    guardrail = result["guardrail"]
    confidence = result["confidence"]

    if not guardrail["passed"]:
        print(f"⚠️  Guardrail Blocked: {answer}")
    else:
        print(f"💡 Answer:\n{answer}")

    # Confidence
    conf_score = confidence["confidence_score"]
    if conf_score >= 0.7:
        conf_emoji, conf_label = "🟢", "High"
    elif conf_score >= 0.4:
        conf_emoji, conf_label = "🟡", "Medium"
    else:
        conf_emoji, conf_label = "🔴", "Low"

    print()
    print(f"{conf_emoji} Confidence: {conf_score:.1%} ({conf_label})")
    print(f"   Retrieval Similarity: {confidence['retrieval_similarity']:.1%}")
    print(f"   Chunk Agreement:      {confidence['chunk_agreement']:.1%}")
    print(f"   Answer Coverage:      {confidence['answer_coverage']:.1%}")

    # Sources
    if result["sources"]:
        print()
        print("📄 Sources:")
        for i, src in enumerate(result["sources"], 1):
            snippet = src["text"][:200].replace("\n", " ")
            print(f"   [{i}] Page {src['page']} (sim: {src['similarity']:.2f}) — {snippet}...")

    print_separator()


def run_extraction(file_path: str):
    """Extract structured data from the document."""
    print_separator()
    print("⚡ Extracting structured shipment data...")
    print()

    result = extract_shipment_data(file_path=file_path)

    error = result.pop("_error", None)
    if error:
        print(f"⚠️  Warning: {error}")
        print()

    print(json.dumps(result, indent=2, default=str))
    print_separator()


# ---------------------------------------------------------------------------
#  Main — Interactive CLI
# ---------------------------------------------------------------------------

def main():
    print_banner()

    # Accept document path from command line or prompt
    if len(sys.argv) >= 2:
        file_path = sys.argv[1]
    else:
        print("Supported formats: PDF, DOCX, TXT, PNG, JPG")
        file_path = input("📁 Enter document path: ").strip().strip('"').strip("'")

    # Process the document
    doc_id, save_path = upload_and_process(file_path)

    # Interactive Q&A loop
    print("=" * 60)
    print("  Ready! Type your questions below.")
    print("  Commands:")
    print("    extract  — Extract structured shipment data as JSON")
    print("    quit     — Exit")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input("❓ Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("\n👋 Goodbye!")
            break

        if user_input.lower() == "extract":
            run_extraction(save_path)
            continue

        # Normal Q&A
        run_qa(doc_id, user_input)
        print()


if __name__ == "__main__":
    main()
