"""
Guardrails and confidence scoring module.
Implements hallucination prevention and multi-factor confidence scoring.
"""

from typing import List, Tuple
from backend.config import config


def compute_similarity_score(search_results: List[Tuple[str, float, dict]]) -> float:
    """
    Compute the average similarity score from the top-K retrieved chunks.
    Returns value in [0, 1].
    """
    if not search_results:
        return 0.0
    scores = [sim for _, sim, _ in search_results]
    return sum(scores) / len(scores)


def compute_agreement_score(search_results: List[Tuple[str, float, dict]]) -> float:
    """
    Compute chunk agreement: how consistently do top chunks relate to each other?
    High agreement = multiple chunks corroborate the same information.
    
    We use the ratio of chunks above the similarity threshold as a proxy.
    """
    if not search_results:
        return 0.0

    threshold = config.similarity_threshold
    above = sum(1 for _, sim, _ in search_results if sim >= threshold)
    return above / len(search_results)


def compute_coverage_score(answer: str, source_texts: List[str]) -> float:
    """
    Compute answer coverage: what fraction of answer words appear in source texts?
    This helps detect hallucination — answers with low coverage are likely fabricated.
    """
    if not answer or not source_texts:
        return 0.0

    # Normalize
    answer_words = set(answer.lower().split())
    source_combined = " ".join(source_texts).lower()
    source_words = set(source_combined.split())

    if not answer_words:
        return 0.0

    # Remove common stop words from consideration
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "out", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where",
        "why", "how", "all", "both", "each", "few", "more", "most",
        "other", "some", "such", "no", "nor", "not", "only", "own",
        "same", "so", "than", "too", "very", "just", "because", "but",
        "and", "or", "if", "while", "that", "this", "it", "i", "my",
        "your", "his", "her", "its", "our", "their", "what", "which",
        "who", "whom", "these", "those",
    }

    meaningful_answer_words = answer_words - stop_words
    if not meaningful_answer_words:
        return 1.0  # Only stop words → trivially covered

    covered = sum(1 for w in meaningful_answer_words if w in source_words)
    return covered / len(meaningful_answer_words)


def compute_confidence(
    search_results: List[Tuple[str, float, dict]],
    answer: str,
) -> dict:
    """
    Compute a weighted confidence score from multiple factors.
    Returns dict with individual scores and final weighted confidence.
    """
    source_texts = [text for text, _, _ in search_results]

    sim_score = compute_similarity_score(search_results)
    agree_score = compute_agreement_score(search_results)
    cov_score = compute_coverage_score(answer, source_texts)

    weights = config.confidence_weights
    final = (
        weights["similarity"] * sim_score
        + weights["agreement"] * agree_score
        + weights["coverage"] * cov_score
    )

    return {
        "retrieval_similarity": round(sim_score, 3),
        "chunk_agreement": round(agree_score, 3),
        "answer_coverage": round(cov_score, 3),
        "confidence_score": round(final, 3),
    }


def apply_guardrails(
    search_results: List[Tuple[str, float, dict]],
    confidence_info: dict,
) -> dict:
    """
    Apply hallucination guardrails. Returns a dict with:
    - passed: bool — whether the answer should be shown
    - reason: str — explanation if blocked
    """
    # Guardrail 1: No relevant context found at all
    if not search_results:
        return {
            "passed": False,
            "reason": "Not found in document — no relevant context was retrieved.",
        }

    # Guardrail 2: Top result below similarity threshold
    top_similarity = max(sim for _, sim, _ in search_results)
    if top_similarity < config.similarity_threshold:
        return {
            "passed": False,
            "reason": (
                f"Not found in document — retrieval similarity too low "
                f"(best: {top_similarity:.2f}, threshold: {config.similarity_threshold:.2f})."
            ),
        }

    # Guardrail 3: Overall confidence too low
    if confidence_info["confidence_score"] < config.min_confidence_to_answer:
        return {
            "passed": False,
            "reason": (
                f"Confidence too low to provide a reliable answer "
                f"(score: {confidence_info['confidence_score']:.2f}, "
                f"minimum: {config.min_confidence_to_answer:.2f})."
            ),
        }

    return {"passed": True, "reason": ""}
