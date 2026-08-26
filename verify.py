import numpy as np
from sentence_transformers import CrossEncoder

from ingest import ingest_pdf
from retrieve import retrieve_top_chunks


_nli_model = CrossEncoder("cross-encoder/nli-distilroberta-base")
LABELS = ["contradiction", "entailment", "neutral"]


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - np.max(logits))
    return exp / exp.sum()


def classify_chunk(chunk: str, claim: str) -> dict:
    """Run NLI on a single (premise=chunk, hypothesis=claim) pair."""
    logits = _nli_model.predict([(chunk, claim)])[0]
    probs = _softmax(logits)

    label_probs = dict(zip(LABELS, probs))
    predicted_label = max(label_probs, key=label_probs.get)

    return {
        "chunk": chunk,
        "label": predicted_label,                     # still tracked, for display/debugging
        "probability": float(label_probs[predicted_label]),  # still tracked, for display/debugging
        "all_probs": {k: float(v) for k, v in label_probs.items()},  # now actually used downstream
    }


def aggregate_verdict(
    classified_chunks: list[dict],
    retrieval_scores: list[float],
    min_total_weight: float = 0.5,
) -> str:
    """
    Weighted-voting aggregation, full-split version:
    each chunk contributes to ALL THREE label buckets,
    weighted by (that label's probability * chunk relevance).
    The label with the highest total weight wins,
    as long as it clears min_total_weight.
    """
    weighted_totals = {"contradiction": 0.0, "entailment": 0.0, "neutral": 0.0}

    for result, relevance in zip(classified_chunks, retrieval_scores):
        for label in LABELS:
            label_prob = result["all_probs"][label]
            weighted_totals[label] += label_prob * relevance

    top_label = max(weighted_totals, key=weighted_totals.get)
    top_weight = weighted_totals[top_label]

    if top_label == "neutral" or top_weight < min_total_weight:
        return "NOT ENOUGH EVIDENCE"
    elif top_label == "contradiction":
        return "CONTRADICTED"
    else:
        return "SUPPORTED"


def verify_claim(claim: str, chunks: list[str], top_k: int = 5) -> dict:
    """Full Stage 3 pipeline: claim + all chunks -> verdict + evidence."""

    top_chunks = retrieve_top_chunks(claim, chunks, top_k=top_k)
    retrieval_scores = [score for _chunk, score in top_chunks]

    classified = [
        classify_chunk(chunk, claim) for chunk, _score in top_chunks
    ]

    verdict = aggregate_verdict(classified, retrieval_scores)

    return {
        "claim": claim,
        "verdict": verdict,
        "evidence": classified,
        "retrieval_scores": retrieval_scores,
    }


if __name__ == "__main__":
    pdf_path = "sample_cited_paper.pdf"
    chunks = ingest_pdf(pdf_path)

    claim = "The alloy shows a yield strength of 450 MPa after annealing."

    result = verify_claim(claim, chunks, top_k=5)

    print(f"Claim: {result['claim']}")
    print(f"Verdict: {result['verdict']}\n")

    print("Evidence:")
    for item, score in zip(result["evidence"], result["retrieval_scores"]):
        print(f"- [{item['label']} (prob={item['probability']:.2f}), relevance={score:.2f}] {item['chunk'][:200]}...")