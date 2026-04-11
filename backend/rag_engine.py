"""
RAG Engine module.
Retrieves relevant chunks, constructs prompts, queries Gemma 3 4B (GGUF),
and applies guardrails + confidence scoring.

Backend: Gemma 3 4B via llama-cpp-python (local GGUF file).
"""

import os
import re
from typing import Optional

from backend.config import config
from backend.embedding_store import embedding_store
from backend.guardrails import compute_confidence, apply_guardrails


# ---------------------------------------------------------------------------
#  Gemma 3 4B GGUF — Lazy-loaded singleton
# ---------------------------------------------------------------------------
_llm = None


def _load_model():
    """Load Gemma 3 4B GGUF model (once, lazily)."""
    global _llm
    if _llm is not None:
        return _llm

    from llama_cpp import Llama

    model_path = os.path.abspath(config.model_path)
    print(f"\nLoading Gemma 3 4B from: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Place gemma-3-4b-it-Q4_K_M.gguf in the models/ directory."
        )

    _llm = Llama(
        model_path=model_path,
        n_ctx=config.n_ctx,
        n_gpu_layers=config.n_gpu_layers,
        verbose=False,
    )
    print("Gemma 3 4B loaded successfully!\n")
    return _llm


# ---------------------------------------------------------------------------
#  LLM Call
# ---------------------------------------------------------------------------

def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Call Gemma 3 4B locally via llama-cpp-python."""
    llm = _load_model()

    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=config.max_tokens,
        temperature=0.2,
        top_p=0.9,
    )

    raw_answer = response["choices"][0]["message"]["content"].strip()
    return raw_answer.replace("\\n", "\n")


# ---------------------------------------------------------------------------
#  RAG Q&A
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = """You are a precise document analysis assistant for logistics, warehouse, and supply-chain documents.
You handle documents such as pick lists, invoices, bills of lading, purchase orders, and shipping manifests.

STRICT RULES:
1. ONLY answer based on the provided context below. Do NOT use any outside knowledge.
2. If the answer is not found in the context, say: "Not found in the provided document."
3. Quote relevant text from the context as supporting evidence.
4. Be concise and factual.
5. If a question is ambiguous, state what you found and note the ambiguity.
6. For tabular data (items, quantities, SKUs), present answers in a clear list or table format.
7. If asked about totals or counts, compute them from the data found in the context.
8. Pay attention to structured fields like [HEADER], [ROW], SKU codes, quantities, and pipe-delimited table data."""

RAG_USER_TEMPLATE = """CONTEXT (retrieved from the document):
---
{context}
---

QUESTION: {question}

Provide a clear, grounded answer based ONLY on the context above. Include the supporting source text."""

SUMMARY_SYSTEM_PROMPT = """You are a helpful logistics document assistant. 
The user is asking for a general overview or summary of the document.
Based on the provided chunks from the beginning of the document, explain what this document is about.
Be concise, factual, and DO NOT fabricate any details."""

SUMMARY_USER_TEMPLATE = """CONTEXT (beginning of the document):
---
{context}
---

QUESTION: {question}

Provide a concise, high-level summary of what this document is about based on the context above."""


def ask_question(doc_id: str, question: str) -> dict:
    """
    Full RAG pipeline: retrieve → build prompt → call LLM → score → guardrail.

    Returns:
        {
            "answer": str,
            "sources": [{"text": str, "similarity": float, "page": int}],
            "confidence": {...},
            "guardrail": {"passed": bool, "reason": str},
        }
    """
    lower_q = question.lower()
    summary_keywords = ["what is this", "what's this", "summarize", "summary", "overview", "about", "what is the doc"]
    is_summary = any(kw in lower_q for kw in summary_keywords)

    # 1. Retrieve relevant chunks
    if is_summary:
        search_results = embedding_store.get_first_chunks(doc_id, k=3)
    else:
        search_results = embedding_store.search(doc_id, question)

    # 2. Format sources for output
    sources = [
        {
            "text": text,
            "similarity": round(sim, 3),
            "page": meta.get("page", 0),
        }
        for text, sim, meta in search_results
    ]

    # 3. Build context from retrieved chunks
    context_parts = []
    for i, (text, sim, meta) in enumerate(search_results, 1):
        context_parts.append(f"[Source {i} | Page {meta.get('page', '?')} | Similarity: {sim:.2f}]\n{text}")
    context = "\n\n".join(context_parts) if context_parts else "No relevant context found."

    # 4. Early guardrail check (before calling LLM, to save compute)
    preliminary_confidence = compute_confidence(search_results, "")
    preliminary_guardrail = apply_guardrails(search_results, preliminary_confidence)

    if not preliminary_guardrail["passed"]:
        return {
            "answer": preliminary_guardrail["reason"],
            "sources": sources,
            "confidence": preliminary_confidence,
            "guardrail": preliminary_guardrail,
        }

    # 5. Call LLM
    if is_summary:
        user_prompt = SUMMARY_USER_TEMPLATE.format(context=context, question=question)
        system_prompt = SUMMARY_SYSTEM_PROMPT
    else:
        user_prompt = RAG_USER_TEMPLATE.format(context=context, question=question)
        system_prompt = RAG_SYSTEM_PROMPT

    try:
        answer = call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    except Exception as e:
        return {
            "answer": f"Error calling LLM: {str(e)}",
            "sources": sources,
            "confidence": preliminary_confidence,
            "guardrail": {"passed": False, "reason": f"LLM call failed: {str(e)}"},
        }

    # 6. Post-answer confidence scoring
    confidence = compute_confidence(search_results, answer)
    guardrail = apply_guardrails(search_results, confidence)

    # If post-answer guardrail fails, override with guardrail message
    if not guardrail["passed"]:
        answer = guardrail["reason"]

    # Clean up internal parsing markers before displaying to user
    final_answer = re.sub(r'\[HEADER\]\s*', '', answer)
    final_answer = re.sub(r'\[ROW \d+\]\s*', '', final_answer)

    return {
        "answer": final_answer,
        "sources": sources,
        "confidence": confidence,
        "guardrail": guardrail,
    }
