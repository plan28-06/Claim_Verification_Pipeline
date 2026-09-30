import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

NLI_THRESHOLD = 0.52
TOP_K = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Number of claim/evidence pairs processed together.
# 32 is a safe starting point for an RTX 4080.
NLI_BATCH_SIZE = 32


# ============================================================
# LOAD MODEL
# ============================================================

_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

_nli_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME
)

_nli_model.to(DEVICE)
_nli_model.eval()


_id2label = {
    int(k): v.lower()
    for k, v in _nli_model.config.id2label.items()
}


# ============================================================
# SINGLE-PAIR CLASSIFICATION
# ============================================================

def classify_chunk(chunk: str, claim: str) -> dict:

    inputs = _tokenizer(
        chunk,
        claim,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        logits = _nli_model(**inputs).logits[0]

    probs = torch.softmax(logits, dim=0)

    label_probs = {
        _id2label[i]: float(probs[i].cpu())
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
# BATCH CLASSIFICATION
# ============================================================

def classify_chunks_batch(
    chunks: list[tuple[str, str]]
) -> list[dict]:

    results = []

    for start in range(
        0,
        len(chunks),
        NLI_BATCH_SIZE
    ):

        batch = chunks[
            start:start + NLI_BATCH_SIZE
        ]

        texts = [
            item[0]
            for item in batch
        ]

        claims = [
            item[1]
            for item in batch
        ]

        inputs = _tokenizer(
            texts,
            claims,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        with torch.inference_mode():

            logits = _nli_model(
                **inputs
            ).logits

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        for j in range(len(batch)):

            probs = probabilities[j]

            label_probs = {
                _id2label[i]:
                    float(probs[i].cpu())
                for i in range(len(probs))
            }

            predicted_label = max(
                label_probs,
                key=label_probs.get
            )

            results.append({
                "chunk": batch[j][0],
                "label": predicted_label,
                "probability":
                    label_probs[predicted_label],
                "all_probs": label_probs,
            })

    return results


# ============================================================
# NLI EVIDENCE AGGREGATION
# ============================================================

def aggregate_nli_evidence(
    classified_chunks,
    k=TOP_K,
    threshold=NLI_THRESHOLD
):

    chunks = sorted(
        classified_chunks,
        key=lambda item:
            item["retrieval_score"],
        reverse=True
    )[:k]

    if not chunks:

        return {
            "verdict": "NOT ENOUGH EVIDENCE",
            "support_score": 0.0,
            "contradiction_score": 0.0,
            "neutral_score": 0.0,
            "support_ratio": 0.0,
            "contradiction_ratio": 0.0,
            "neutral_ratio": 0.0,
            "support_vs_contradiction": 0.0,
            "contradiction_vs_support": 0.0,
        }

    support_score = 0.0
    contradiction_score = 0.0
    neutral_score = 0.0

    for item in chunks:

        relevance = float(
            item["retrieval_score"]
        )

        probabilities = item["all_probs"]

        support_score += (
            relevance *
            probabilities["entailment"]
        )

        contradiction_score += (
            relevance *
            probabilities["contradiction"]
        )

        neutral_score += (
            relevance *
            probabilities["neutral"]
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
            "support_ratio": 0.0,
            "contradiction_ratio": 0.0,
            "neutral_ratio": 0.0,
            "support_vs_contradiction": 0.0,
            "contradiction_vs_support": 0.0,
        }

    support_ratio = (
        support_score / total_score
    )

    contradiction_ratio = (
        contradiction_score / total_score
    )

    neutral_ratio = (
        neutral_score / total_score
    )

    non_neutral = (
        support_score +
        contradiction_score
    )

    if non_neutral == 0:

        verdict = "NOT ENOUGH EVIDENCE"

        support_vs_contradiction = 0.0
        contradiction_vs_support = 0.0

    else:

        support_vs_contradiction = (
            support_score / non_neutral
        )

        contradiction_vs_support = (
            contradiction_score /
            non_neutral
        )

        if (
            support_vs_contradiction
            >= threshold
        ):

            verdict = "SUPPORTED"

        elif (
            contradiction_vs_support
            >= threshold
        ):

            verdict = "CONTRADICTED"

        else:

            verdict = "NOT ENOUGH EVIDENCE"

    return {
        "verdict": verdict,
        "support_score": support_score,
        "contradiction_score":
            contradiction_score,
        "neutral_score": neutral_score,
        "support_ratio": support_ratio,
        "contradiction_ratio":
            contradiction_ratio,
        "neutral_ratio": neutral_ratio,
        "support_vs_contradiction":
            support_vs_contradiction,
        "contradiction_vs_support":
            contradiction_vs_support,
    }


# ============================================================
# CLAIM VERIFICATION
# ============================================================

def verify_claim_test(
    claim: str,
    retrieved_evidence:
        list[tuple[str, float]]
) -> dict:

    # Prepare pairs
    pairs = [
        (sentence, claim)
        for sentence, _ in retrieved_evidence
    ]

    # Batch inference
    classified = classify_chunks_batch(
        pairs
    )

    # Restore retrieval scores
    for result, (_, retrieval_score) in zip(
        classified,
        retrieved_evidence
    ):

        result["retrieval_score"] = float(
            retrieval_score
        )

    # Frozen aggregation
    aggregation = aggregate_nli_evidence(
        classified,
        k=TOP_K,
        threshold=NLI_THRESHOLD
    )

    return {
        "claim": claim,
        "verdict":
            aggregation["verdict"],

        "support_score":
            aggregation["support_score"],

        "contradiction_score":
            aggregation["contradiction_score"],

        "neutral_score":
            aggregation["neutral_score"],

        "support_ratio":
            aggregation["support_ratio"],

        "contradiction_ratio":
            aggregation["contradiction_ratio"],

        "neutral_ratio":
            aggregation["neutral_ratio"],

        "support_vs_contradiction":
            aggregation[
                "support_vs_contradiction"
            ],

        "contradiction_vs_support":
            aggregation[
                "contradiction_vs_support"
            ],

        "evidence": classified,
    }


# ============================================================
# DEVICE CHECK
# ============================================================

print(
    f"NLI device: {DEVICE}"
)

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )