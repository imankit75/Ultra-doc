"""
RAG Engine module.
Retrieves relevant chunks, constructs prompts, queries the LLM via
HuggingFace Transformers, and applies guardrails + confidence scoring.

Device adaptation (auto-detected via config.device):
  cuda  — NF4 4-bit quantization via bitsandbytes  (~3 GB VRAM)
  mps   — float16, no quantization (Apple Silicon)  (~4 GB unified memory)
  cpu   — int8 weight-only quantization via torchao (~5 GB RAM)
"""

import re
import time

import torch

from backend.config import config
from backend.embedding_store import embedding_store
from backend.guardrails import compute_confidence, apply_guardrails


_model = None
_tokenizer = None


def _load_model():
    """Load the LLM once (lazy singleton). Quantization and device adapt automatically.

    Loading order:
      1. Local GGUF file  — used when HF_TOKEN is absent and GGUF file exists on disk.
      2. HuggingFace Hub  — used when HF_TOKEN env var is set (gated/latest models).
    """
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    import os as _os
    from transformers import AutoTokenizer, AutoModelForCausalLM

    gguf_path = _os.path.join(config.local_model_dir, config.gguf_file)
    hf_token = _os.getenv("HF_TOKEN")
    use_local = not hf_token and _os.path.exists(gguf_path)

    print("\n" + "="*60)
    print("  MODEL LOADING")
    print("="*60)

    if use_local:
        print(f"  Source  : LOCAL GGUF")
        print(f"  Path    : {gguf_path}")
        print(f"  Device  : {config.device.upper()}")
        print("="*60)
        load_kwargs = dict(
            pretrained_model_name_or_path=config.local_model_dir,
            gguf_file=config.gguf_file,
        )
        print("[1/3] Loading tokenizer from local GGUF...")
        _tokenizer = AutoTokenizer.from_pretrained(
            config.local_model_dir, gguf_file=config.gguf_file
        )
    else:
        print(f"  Source  : HuggingFace Hub")
        print(f"  Model   : {config.model_id}")
        print(f"  Device  : {config.device.upper()}")
        print(f"  HF token: {'SET' if hf_token else 'NOT SET (public model only)'}")
        print("="*60)
        load_kwargs = dict(pretrained_model_name_or_path=config.model_id)
        print("[1/3] Loading tokenizer from HF Hub...")
        _tokenizer = AutoTokenizer.from_pretrained(config.model_id, token=hf_token)

    print("      Tokenizer loaded.\n")

    # -----------------------------------------------------------------------
    # CUDA — NF4 4-bit via bitsandbytes
    # -----------------------------------------------------------------------
    if config.device == "cuda":
        from transformers import BitsAndBytesConfig

        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        # Reserve 20% headroom for KV cache, minimum 1 GB
        max_vram = max(1.0, vram_total * 0.80)

        print("[2/3] Loading model with bitsandbytes NF4 4-bit quantization...")
        print(f"      GPU VRAM : {vram_total:.1f} GB total  ({max_vram:.1f} GB reserved for model)")
        print("      (first run downloads model from HF Hub — may take a few minutes)")
        t0 = time.time()

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        try:
            _model = AutoModelForCausalLM.from_pretrained(
                **load_kwargs,
                quantization_config=bnb_config,
                device_map={"": 0},
                max_memory={0: f"{max_vram:.0f}GiB"},
                token=hf_token if not use_local else None,
            )
        except torch.cuda.OutOfMemoryError:
            print("\n  WARNING: GPU out of memory — falling back to CPU (int8).")
            torch.cuda.empty_cache()
            config.device = "cpu"
            _model = _load_cpu(load_kwargs, hf_token, use_local)

        if config.device == "cuda":
            print(f"      Model loaded + quantized in {time.time() - t0:.1f}s.\n")
            print("[3/3] Verifying GPU placement...")
            vram_used = torch.cuda.memory_allocated() / 1024**3
            vram_free = vram_total - vram_used
            print(f"      GPU   : {torch.cuda.get_device_name(0)}")
            print(f"      VRAM  : {vram_used:.1f} GB used / {vram_total:.1f} GB total  ({vram_free:.1f} GB free)")

    # -----------------------------------------------------------------------
    # MPS — Apple Silicon (float16, no quantization)
    # -----------------------------------------------------------------------
    elif config.device == "mps":
        print("[2/3] Loading model for Apple Silicon (MPS)...")
        print("      Using float16 — bitsandbytes not supported on MPS")
        print("      (first run downloads model from HF Hub — may take a few minutes)")
        t0 = time.time()

        _model = AutoModelForCausalLM.from_pretrained(
            **load_kwargs,
            torch_dtype=torch.float16,
            device_map="mps",
            token=hf_token if not use_local else None,
        )
        print(f"      Model loaded in {time.time() - t0:.1f}s.\n")
        print("[3/3] Verifying MPS placement...")
        print(f"      Device: Apple Silicon MPS")

    # -----------------------------------------------------------------------
    # CPU — int8 weight-only via torchao
    # -----------------------------------------------------------------------
    else:
        _model = _load_cpu(load_kwargs, hf_token, use_local)

    print(f"\n  [OK] Model ready on {config.device.upper()}!")
    print("="*60 + "\n")
    return _model, _tokenizer


def _load_cpu(load_kwargs: dict, hf_token, use_local: bool):
    """Load model on CPU with torchao int8 weight-only quantization."""
    from transformers import AutoModelForCausalLM
    from torchao.quantization import quantize_, Int8WeightOnlyConfig

    print("[2/3] Loading model weights to CPU in bfloat16...")
    print("      (first run downloads model from HF Hub — may take a few minutes)")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        **load_kwargs,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        token=hf_token if not use_local else None,
    )
    print(f"      Weights loaded in {time.time() - t0:.1f}s.\n")

    print("[3/3] Applying int8 weight-only quantization...")
    t0 = time.time()
    try:
        quantize_(model, Int8WeightOnlyConfig())
        print(f"      Quantization done in {time.time() - t0:.1f}s.")
    except Exception as e:
        print(f"      WARN: torchao quantization failed ({e}) — using bfloat16 (higher RAM).")

    return model


def call_llm(system_prompt: str, user_prompt: str, max_tokens_override: int = None) -> str:
    """Call the LLM locally via HuggingFace Transformers generate()."""
    model, tokenizer = _load_model()

    max_new_tokens = max_tokens_override if max_tokens_override else config.max_new_tokens

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_token_count = inputs["input_ids"].shape[1]

    print(f"\n[LLM] Inference starting...")
    print(f"      Prompt tokens : {input_token_count}")
    print(f"      Max new tokens: {max_new_tokens}")
    print(f"      Device        : {model.device}")
    t0 = time.time()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
        )

    elapsed = time.time() - t0
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    output_token_count = new_tokens.shape[0]
    tps = output_token_count / elapsed if elapsed > 0 else 0

    print(f"      Output tokens : {output_token_count}")
    print(f"      Time          : {elapsed:.1f}s  ({tps:.1f} tok/s)\n")

    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


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

    mode = "SUMMARY" if is_summary else "Q&A"
    print(f"\n[RAG] Question received ({mode}): {question[:80]}{'...' if len(question) > 80 else ''}")

    print(f"[RAG] Retrieving chunks from vector store...")
    if is_summary:
        search_results = embedding_store.get_first_chunks(doc_id, k=3)
    else:
        search_results = embedding_store.search(doc_id, question)

    print(f"[RAG] Retrieved {len(search_results)} chunk(s).")
    if search_results:
        print(f"      Top similarity: {search_results[0][1]:.3f}")

    sources = [
        {
            "text": text,
            "similarity": round(sim, 3),
            "page": meta.get("page", 0),
        }
        for text, sim, meta in search_results
    ]

    context_parts = []
    for i, (text, sim, meta) in enumerate(search_results, 1):
        context_parts.append(f"[Source {i} | Page {meta.get('page', '?')} | Similarity: {sim:.2f}]\n{text}")
    context = "\n\n".join(context_parts) if context_parts else "No relevant context found."

    preliminary_confidence = compute_confidence(search_results, "")
    preliminary_guardrail = apply_guardrails(search_results, preliminary_confidence)

    if not preliminary_guardrail["passed"]:
        print(f"[RAG] Guardrail blocked (pre-LLM): {preliminary_guardrail['reason']}")
        return {
            "answer": preliminary_guardrail["reason"],
            "sources": sources,
            "confidence": preliminary_confidence,
            "guardrail": preliminary_guardrail,
        }

    if is_summary:
        user_prompt = SUMMARY_USER_TEMPLATE.format(context=context, question=question)
        system_prompt = SUMMARY_SYSTEM_PROMPT
    else:
        user_prompt = RAG_USER_TEMPLATE.format(context=context, question=question)
        system_prompt = RAG_SYSTEM_PROMPT

    try:
        answer = call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
    except Exception as e:
        return {
            "answer": f"Error calling LLM: {str(e)}",
            "sources": sources,
            "confidence": preliminary_confidence,
            "guardrail": {"passed": False, "reason": f"LLM call failed: {str(e)}"},
        }

    confidence = compute_confidence(search_results, answer)
    guardrail = apply_guardrails(search_results, confidence)

    if not guardrail["passed"]:
        print(f"[RAG] Guardrail blocked (post-LLM): {guardrail['reason']}")
        answer = guardrail["reason"]

    conf_score = confidence["confidence_score"]
    print(f"[RAG] Confidence: {conf_score:.1%} | Guardrail: {'PASS' if guardrail['passed'] else 'BLOCK'}\n")

    final_answer = re.sub(r'\[HEADER\]\s*', '', answer)
    final_answer = re.sub(r'\[ROW \d+\]\s*', '', final_answer)

    return {
        "answer": final_answer,
        "sources": sources,
        "confidence": confidence,
        "guardrail": guardrail,
    }
