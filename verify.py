import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ingest import ingest_pdf
from retrieve import retrieve_top_chunks


MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_nli_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
_nli_model.eval()

# Read the label order directly from the model's own config —
# no more manual guessing/hardcoding, so this can't silently mismatch.
_id2label = {int(k): v.lower() for k, v in _nli_model.config.id2label.items()}


def classify_chunk(chunk: str, claim: str) -> dict:
    """Run NLI on a single (premise=chunk, hypothesis=claim) pair."""
    inputs = _tokenizer(chunk, claim, return_tensors="pt", truncation=True)

    with torch.no_grad():
        logits = _nli_model(**inputs).logits[0]

    probs = torch.softmax(logits, dim=0)

    label_probs = {_id2label[i]: float(probs[i]) for i in range(len(probs))}
    predicted_label = max(label_probs, key=label_probs.get)

    return {
        "chunk": chunk,
        "label": predicted_label,
        "probability": label_probs[predicted_label],
        "all_probs": label_probs,
    }


def verify_claim(claim: str, chunks: list[str], top_k: int = 3) -> dict:
    """
    Retrieve top_k chunks for context/display,
    but the verdict is decided by the single most relevant chunk only.
    """
    top_chunks = retrieve_top_chunks(claim, chunks, top_k=top_k)
    retrieval_scores = [score for _chunk, score in top_chunks]

    classified = [
        classify_chunk(chunk, claim) for chunk, _score in top_chunks
    ]

    best = classified[0]  # highest-relevance chunk decides the verdict
    label_to_verdict = {
        "entailment": "SUPPORTED",
        "contradiction": "CONTRADICTED",
        "neutral": "NOT ENOUGH EVIDENCE",
    }
    verdict = label_to_verdict[best["label"]]

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

    result = verify_claim(claim, chunks, top_k=3)

    print(f"Claim: {result['claim']}")
    print(f"Verdict: {result['verdict']}\n")

    print("Evidence:")
    for item, score in zip(result["evidence"], result["retrieval_scores"]):
        print(f"- [{item['label']} (prob={item['probability']:.2f}), relevance={score:.2f}] {item['chunk'][:200]}...")
        print(f"    all_probs: {item['all_probs']}")