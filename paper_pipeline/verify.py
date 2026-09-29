import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ingest import ingest_pdf
from retrieve import retrieve_top_chunks


# ============================================================
# NLI MODEL
# ============================================================

MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_nli_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
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
    Run NLI on one paper chunk.

    Premise     = paper chunk
    Hypothesis = claim

    Returns:
        entailment probability
        contradiction probability
        neutral probability
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

    support_score = 0.0
    contradiction_score = 0.0
    neutral_score = 0.0

    for item in classified_chunks:

        relevance = item["retrieval_score"]

        support_score += (
            relevance * item["all_probs"]["entailment"]
        )

        contradiction_score += (
            relevance * item["all_probs"]["contradiction"]
        )

        neutral_score += (
            relevance * item["all_probs"]["neutral"]
        )

    total_score = (
        support_score
        + contradiction_score
        + neutral_score
    )

    if total_score == 0:
        return {
            "verdict": "NOT ENOUGH EVIDENCE",
            "support_score": 0.0,
            "contradiction_score": 0.0,
            "neutral_score": 0.0,
        }

    support_ratio = support_score / total_score
    contradiction_ratio = contradiction_score / total_score
    neutral_ratio = neutral_score / total_score

    # Compare actual positive vs negative evidence.
    non_neutral = support_score + contradiction_score

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
# VERIFY CLAIM
# ============================================================

def verify_claim(
    claim: str,
    chunks: list[str],
    top_k: int = 5
) -> dict:
    """
    Full NLI verification pipeline:

        Claim
          ↓
        Hybrid retrieval
          ↓
        Top-K chunks
          ↓
        NLI classification
          ↓
        Weighted aggregation
          ↓
        NLI verdict
    """

    # --------------------------------------------------------
    # 1. Retrieve relevant chunks
    # --------------------------------------------------------

    top_chunks = retrieve_top_chunks(
        claim,
        chunks,
        top_k=top_k
    )

    # --------------------------------------------------------
    # 2. Classify every retrieved chunk
    # --------------------------------------------------------

    classified = []

    for chunk, retrieval_score in top_chunks:

        result = classify_chunk(
            chunk,
            claim
        )

        # Keep retrieval relevance so that it can
        # be used during aggregation.
        result["retrieval_score"] = float(
            retrieval_score
        )

        classified.append(result)

    # --------------------------------------------------------
    # 3. Aggregate NLI evidence
    # --------------------------------------------------------

    aggregation = aggregate_nli_evidence(
        classified
    )

    # --------------------------------------------------------
    # 4. Return complete result
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


