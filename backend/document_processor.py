"""
Document processing module.
Handles parsing PDF, DOCX, TXT files and intelligent text chunking.
"""

import os
import re
from dataclasses import dataclass
from typing import List

import fitz  # PyMuPDF
from docx import Document as DocxDocument

from backend.config import config

# Use local tessdata if present, otherwise fall back to system tesseract install
_local_tessdata = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tessdata"))
if os.path.isdir(_local_tessdata):
    os.environ["TESSDATA_PREFIX"] = _local_tessdata


@dataclass
class DocumentChunk:
    """A chunk of text extracted from a document."""
    text: str
    chunk_id: int
    source_file: str
    page: int = 0  # 0-based; only meaningful for PDF
    start_char: int = 0


def parse_pdf(file_path: str) -> List[dict]:
    """Extract text from a PDF file, page by page."""
    pages = []
    doc = fitz.open(file_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        if text.strip():
            pages.append({"text": text, "page": page_num + 1})
    doc.close()
    return pages


def parse_docx(file_path: str) -> List[dict]:
    """Extract text from a DOCX file with improved table handling."""
    doc = DocxDocument(file_path)
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    # Extract tables with clear structure
    for table_idx, table in enumerate(doc.tables):
        full_text += "\n\n--- TABLE START ---\n"
        for row_idx, row in enumerate(table.rows):
            cells = [cell.text.strip() for cell in row.cells]
            row_text = " | ".join(cells)
            if row_text.strip(" |\t"):
                if row_idx == 0:
                    full_text += f"[HEADER] {row_text}\n"
                else:
                    full_text += f"[ROW {row_idx}] {row_text}\n"
        full_text += "--- TABLE END ---\n"

    return [{"text": full_text, "page": 1}]


def parse_txt(file_path: str) -> List[dict]:
    """Extract text from a TXT file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [{"text": text, "page": 1}]


def parse_image(file_path: str) -> List[dict]:
    """Extract text from an image using Tesseract OCR."""
    try:
        from PIL import Image
        import pytesseract
    except ImportError:
        raise ValueError("Image support requires 'pillow' and 'pytesseract'.")

    try:
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        if not text.strip():
            text = "[No readable text found in image]"
        return [{"text": text, "page": 1}]
    except Exception as e:
        raise RuntimeError(f"Failed to process image: {str(e)}")


def parse_document(file_path: str) -> List[dict]:
    """Parse a document based on its extension. Returns list of {text, page}."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext in [".docx", ".doc"]:
        return parse_docx(file_path)
    elif ext == ".txt":
        return parse_txt(file_path)
    elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
        return parse_image(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported: PDF, DOCX, DOC, TXT, PNG, JPG, JPEG")


def chunk_text(
    text: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[str]:
    """
    Split text into overlapping chunks.
    Preserves table rows and structured blocks together when possible.
    """
    chunk_size = chunk_size or config.chunk_size
    chunk_overlap = chunk_overlap or config.chunk_overlap

    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    # ----- Step 1: Separate table blocks from prose -----
    # Split into segments: prose sections and table sections
    segments = _split_tables_and_prose(text)

    chunks = []
    for segment in segments:
        seg_text = segment.strip()
        if not seg_text:
            continue
        if len(seg_text) <= chunk_size:
            chunks.append(seg_text)
        else:
            # For large segments, use line-aware splitting
            chunks.extend(_split_segment(seg_text, chunk_size, chunk_overlap))

    return chunks


def _split_tables_and_prose(text: str) -> List[str]:
    """Split text into alternating prose and table segments so tables stay intact."""
    segments = []
    current = []
    in_table = False

    for line in text.split("\n"):
        if "--- TABLE START ---" in line:
            # Flush current prose
            if current:
                segments.append("\n".join(current))
                current = []
            in_table = True
            current.append(line)
        elif "--- TABLE END ---" in line:
            current.append(line)
            segments.append("\n".join(current))
            current = []
            in_table = False
        else:
            current.append(line)

    if current:
        segments.append("\n".join(current))

    return segments


def _split_segment(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split a large text segment using paragraph / sentence boundaries."""
    separators = ["\n\n", "\n", ". ", " "]
    chunks = []

    for sep in separators:
        parts = text.split(sep)
        if len(parts) > 1:
            current_chunk = ""
            for part in parts:
                candidate = current_chunk + sep + part if current_chunk else part
                if len(candidate) <= chunk_size:
                    current_chunk = candidate
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    if chunk_overlap > 0 and current_chunk:
                        overlap_text = current_chunk[-chunk_overlap:]
                        current_chunk = overlap_text + sep + part
                    else:
                        current_chunk = part

            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            if chunks:
                return chunks

    # Fallback: hard split
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunk = text[i : i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def process_document(file_path: str) -> List[DocumentChunk]:
    """
    Full pipeline: parse document → chunk text → return DocumentChunk objects.
    """
    filename = os.path.basename(file_path)
    ext = os.path.splitext(file_path)[1].upper()
    print(f"[DOC] Parsing {ext} file: {filename}")

    pages = parse_document(file_path)
    print(f"[DOC] Parsed {len(pages)} page(s). Chunking text...")

    all_chunks = []
    chunk_id = 0

    for page_info in pages:
        text = page_info["text"]
        page = page_info["page"]
        text_chunks = chunk_text(text)

        for chunk_text_str in text_chunks:
            all_chunks.append(DocumentChunk(
                text=chunk_text_str,
                chunk_id=chunk_id,
                source_file=filename,
                page=page,
            ))
            chunk_id += 1

    print(f"[DOC] Created {len(all_chunks)} chunk(s) from {len(pages)} page(s).")
    return all_chunks


def get_full_text(file_path: str) -> str:
    """Get the full concatenated text of a document (used for extraction)."""
    pages = parse_document(file_path)
    return "\n\n".join(p["text"] for p in pages)
