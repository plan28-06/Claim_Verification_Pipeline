import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# NLI MODEL
# ============================================================

MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_nli_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME
)

_nli_model.eval()

# Read label order directly from the model configuration.
_id2label = {
    int(k): v.lower()
    for k, v in _nli_model.config.id2label.items()
}


# ============================================================
# CLASSIFY ONE CHUNK
# ============================================================

def classify_chunk(chunk: str, claim: str) -> dict:
    """
    Run NLI on one paper sentence/chunk.

    Premise     = paper evidence
    Hypothesis  = claim

    Returns:
        predicted label
        predicted probability
        probabilities for entailment,
        contradiction, and neutral
    """

    inputs = _tokenizer(
        chunk,
        claim,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        logits = _nli_model(**inputs).logits[0]

    # Convert logits into probabilities.
    probs = torch.softmax(logits, dim=0)

    label_probs = {
        _id2label[i]: float(probs[i])
        for i in range(len(probs))
    }

    predicted_label = max(
        label_probs,
        key=label_probs.get
    )

    return {
        "chunk": chunk,
        "label": predicted_label,
        "probability": label_probs[predicted_label],
        "all_probs": label_probs,
    }


# ============================================================
# AGGREGATE NLI EVIDENCE
# ============================================================

def aggregate_nli_evidence(classified_chunks):
    """
    Aggregate NLI predictions using retrieval relevance
    as the weighting factor.

    Each classified chunk must contain:

        retrieval_score
        all_probs["entailment"]
        all_probs["contradiction"]
        all_probs["neutral"]
    """

    support_score = 0.0
    contradiction_score = 0.0
    neutral_score = 0.0

    for item in classified_chunks:

        relevance = item["retrieval_score"]

        support_score += (
            relevance
            * item["all_probs"]["entailment"]
        )

        contradiction_score += (
            relevance
            * item["all_probs"]["contradiction"]
        )

        neutral_score += (
            relevance
            * item["all_probs"]["neutral"]
        )

    total_score = (
        support_score
        + contradiction_score
        + neutral_score
    )

    # --------------------------------------------------------
    # No usable evidence
    # --------------------------------------------------------

    if total_score == 0:

        return {
            "verdict": "NOT ENOUGH EVIDENCE",
            "support_score": 0.0,
            "contradiction_score": 0.0,
            "neutral_score": 0.0,
            "support_ratio": 0.0,
            "contradiction_ratio": 0.0,
            "neutral_ratio": 0.0,
        }

    # --------------------------------------------------------
    # Normalize scores
    # --------------------------------------------------------

    support_ratio = (
        support_score / total_score
    )

    contradiction_ratio = (
        contradiction_score / total_score
    )

    neutral_ratio = (
        neutral_score / total_score
    )

    # --------------------------------------------------------
    # Compare positive vs negative evidence
    # --------------------------------------------------------

    non_neutral = (
        support_score
        + contradiction_score
    )

    if non_neutral == 0:

        verdict = "NOT ENOUGH EVIDENCE"

    else:

        support_vs_contradiction = (
            support_score / non_neutral
        )

        contradiction_vs_support = (
            contradiction_score / non_neutral
        )

        if support_vs_contradiction >= 0.65:

            verdict = "SUPPORTED"

        elif contradiction_vs_support >= 0.65:

            verdict = "CONTRADICTED"

        else:

            verdict = "NOT ENOUGH EVIDENCE"

    return {
        "verdict": verdict,

        "support_score": support_score,

        "contradiction_score": contradiction_score,

        "neutral_score": neutral_score,

        "support_ratio": support_ratio,

        "contradiction_ratio": contradiction_ratio,

        "neutral_ratio": neutral_ratio,
    }


# ============================================================
# VERIFY CLAIM USING CACHED EVIDENCE
# ============================================================

def verify_claim_test(
    claim: str,
    retrieved_evidence: list[tuple[str, float]]
) -> dict:
    """
    Run NLI verification on already-retrieved evidence.

    IMPORTANT:
        This function does NOT perform retrieval.

    retrieved_evidence format:

        [
            (sentence, relevance_score),
            ...
        ]

    The retrieval scores come from the frozen
    retrieval_cache.json.
    """

    classified = []

    # --------------------------------------------------------
    # 1. Classify each cached evidence sentence
    # --------------------------------------------------------

    for sentence, retrieval_score in retrieved_evidence:

        result = classify_chunk(
            sentence,
            claim
        )

        # Preserve the frozen retrieval score.
        result["retrieval_score"] = float(
            retrieval_score
        )

        classified.append(result)

    # --------------------------------------------------------
    # 2. Aggregate NLI evidence
    # --------------------------------------------------------

    aggregation = aggregate_nli_evidence(
        classified
    )

    # --------------------------------------------------------
    # 3. Return complete NLI result
    # --------------------------------------------------------

    return {
        "claim": claim,

        "verdict": aggregation["verdict"],

        "support_score": aggregation[
            "support_score"
        ],

        "contradiction_score": aggregation[
            "contradiction_score"
        ],

        "neutral_score": aggregation[
            "neutral_score"
        ],

        "support_ratio": aggregation[
            "support_ratio"
        ],

        "contradiction_ratio": aggregation[
            "contradiction_ratio"
        ],

        "neutral_ratio": aggregation[
            "neutral_ratio"
        ],

        "evidence": classified,
    }