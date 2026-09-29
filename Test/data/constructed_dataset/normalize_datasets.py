import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent
SCIFACT_FILE = BASE_DIR.parent / "scifact" / "scifact_pristine.jsonl"
CLINIFACT_FILE = BASE_DIR.parent / "clinifact" / "clinifact_pristine.jsonl"

SCIFACT_OUTPUT = BASE_DIR / "normalized_scifact.jsonl"
CLINIFACT_OUTPUT = BASE_DIR / "normalized_clinifact.jsonl"


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def split_sentences(text):
    """
    Simple sentence splitting for CliniFact abstracts.
    SciFact already provides sentence-level abstracts.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def normalize_scifact(records):
    normalized = []

    for record in records:

        # SciFact can have multiple cited documents.
        # Keep each claim-document pair as a separate record.
        for document in record["cited_documents"]:

            abstract = document["abstract"]

            # Collect all gold sentence indices
            gold_evidence = []

            for evidence_item in document["evidence"]:
                for idx in evidence_item["sentences"]:
                    if idx not in gold_evidence:
                        gold_evidence.append(idx)

            normalized.append({
                "id": f"scifact_{record['id']}_{document['doc_id']}",
                "source": "SciFact",
                "claim": record["claim"],
                "abstract": abstract,
                "original_label": record["original_label"],
                "label": (
                    "SUPPORTED"
                    if record["original_label"] == "SUPPORT"
                    else "CONTRADICTED"
                ),
                "gold_evidence": sorted(gold_evidence)
            })

    return normalized


def normalize_clinifact(records):
    normalized = []

    for record in records:

        abstract_text = record["abstract"]

        # Convert CliniFact string abstract into sentence list
        abstract = split_sentences(abstract_text)

        normalized.append({
            "id": f"clinifact_{record['id']}",
            "source": "CliniFact",
            "claim": record["claim"],
            "abstract": abstract,
            "original_label": record["original_label"],
            "label": record["label"],
            "gold_evidence": None
        })

    return normalized


def save_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():

    scifact = load_jsonl(SCIFACT_FILE)
    clinifact = load_jsonl(CLINIFACT_FILE)

    normalized_scifact = normalize_scifact(scifact)
    normalized_clinifact = normalize_clinifact(clinifact)

    save_jsonl(normalized_scifact, SCIFACT_OUTPUT)
    save_jsonl(normalized_clinifact, CLINIFACT_OUTPUT)

    print("\nNormalization complete")
    print("-" * 50)

    print(f"SciFact input records:       {len(scifact)}")
    print(f"SciFact normalized records:  {len(normalized_scifact)}")

    print(f"CliniFact input records:     {len(clinifact)}")
    print(f"CliniFact normalized records:{len(normalized_clinifact)}")

    print("-" * 50)

    print(f"Saved: {SCIFACT_OUTPUT}")
    print(f"Saved: {CLINIFACT_OUTPUT}")


if __name__ == "__main__":
    main()