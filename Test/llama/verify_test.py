import sys
from pathlib import Path

# Allow imports from project root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))


from paper_pipeline.verify import (
    classify_chunk,
    aggregate_nli_evidence,
)


def verify_claim_test(
    claim: str,
    retrieved_evidence: list[tuple[str, float]]
) -> dict:
    """
    Run NLI verification on already-retrieved evidence.

    retrieved_evidence:
        [
            (sentence, relevance_score),
            ...
        ]

    This function does NOT perform retrieval.
    """

    classified = []

    for sentence, retrieval_score in retrieved_evidence:

        result = classify_chunk(
            sentence,
            claim
        )

        result["retrieval_score"] = float(
            retrieval_score
        )

        classified.append(result)

    # Aggregate NLI predictions
    aggregation = aggregate_nli_evidence(
        classified
    )

    return {
        "claim": claim,

        "verdict": aggregation["verdict"],

        "support_score": aggregation["support_score"],

        "contradiction_score": aggregation[
            "contradiction_score"
        ],

        "neutral_score": aggregation[
            "neutral_score"
        ],

        "support_ratio": aggregation.get(
            "support_ratio", 0.0
        ),

        "contradiction_ratio": aggregation.get(
            "contradiction_ratio", 0.0
        ),

        "neutral_ratio": aggregation.get(
            "neutral_ratio", 0.0
        ),

        "evidence": classified,
    }