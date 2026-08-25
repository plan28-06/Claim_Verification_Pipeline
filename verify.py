import numpy as np
from sentence_transformers import CrossEncoder

from ingest import ingest_pdf
from retrieve import retrieve_top_chunks


# Load once — reused across calls
_nli_model = CrossEncoder("cross-encoder/nli-distilroberta-base")

# This model's label order — check the model card if you swap models later
LABELS = ["contradiction", "entailment", "neutral"]


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - np.max(logits))  # subtract max for numerical stability
    return exp / exp.sum()


def classify_chunk(chunk: str, claim: str) -> dict:
    """Run NLI on a single (premise=chunk, hypothesis=claim) pair."""
    logits = _nli_model.predict([(chunk, claim)])[0]
    probs = _softmax(logits)

    label_probs = dict(zip(LABELS, probs))
    predicted_label = max(label_probs, key=label_probs.get)

    return {
        "chunk": chunk,
        "label": predicted_label,
        "confidence": float(label_probs[predicted_label]),
        "all_probs": {k: float(v) for k, v in label_probs.items()},
    }


def aggregate_verdict(classified_chunks: list[dict], threshold: float = 0.6) -> str:
    """Combine per-chunk NLI labels into one final verdict."""

    # A single strong contradiction overrides everything else
    for result in classified_chunks:
        if result["label"] == "contradiction" and result["confidence"] >= threshold:
            return "CONTRADICTED"

    # Otherwise, any strong support wins
    for result in classified_chunks:
        if result["label"] == "entailment" and result["confidence"] >= threshold:
            return "SUPPORTED"

    return "NOT ENOUGH EVIDENCE"


def verify_claim(claim: str, chunks: list[str], top_k: int = 5) -> dict:
    """Full Stage 3 pipeline: claim + all chunks -> verdict + evidence."""

    top_chunks = retrieve_top_chunks(claim, chunks, top_k=top_k)

    classified = [
        classify_chunk(chunk, claim) for chunk, _retrieval_score in top_chunks
    ]

    verdict = aggregate_verdict(classified)

    return {
        "claim": claim,
        "verdict": verdict,
        "evidence": classified,
    }


