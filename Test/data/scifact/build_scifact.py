import json
from pathlib import Path

DATA_DIR = Path(__file__).parent

TRAIN_FILE = DATA_DIR / "claims_train.jsonl"
DEV_FILE = DATA_DIR / "claims_dev.jsonl"
CORPUS_FILE = DATA_DIR / "corpus.jsonl"
OUTPUT_FILE = DATA_DIR / "scifact_pristine.jsonl"


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    train = load_jsonl(TRAIN_FILE)
    dev = load_jsonl(DEV_FILE)
    corpus = load_jsonl(CORPUS_FILE)

    claims = train + dev

    corpus_map = {
        str(doc["doc_id"]): doc
        for doc in corpus
    }

    output = []

    stats = {
        "total": len(claims),
        "support": 0,
        "contradict": 0,
        "no_evidence": 0,
        "invalid_source": 0,
        "invalid_evidence": 0
    }

    for claim in claims:

        evidence = claim.get("evidence", {})
        cited_doc_ids = claim.get("cited_doc_ids", [])

        if not evidence:
            stats["no_evidence"] += 1
            continue

        valid_documents = []

        for doc_id in cited_doc_ids:

            doc = corpus_map.get(str(doc_id))

            if doc is None:
                continue

            doc_evidence = evidence.get(str(doc_id), [])

            if not doc_evidence:
                continue

            abstract = doc.get("abstract", [])

            valid_evidence = []

            for item in doc_evidence:

                sentence_ids = item.get("sentences", [])

                if all(
                    isinstance(idx, int)
                    and 0 <= idx < len(abstract)
                    for idx in sentence_ids
                ):
                    valid_evidence.append(item)

            if valid_evidence:
                valid_documents.append({
                    "doc_id": doc_id,
                    "title": doc.get("title", ""),
                    "abstract": abstract,
                    "evidence": valid_evidence
                })

        if not valid_documents:
            stats["invalid_source"] += 1
            continue

        original_label = None

        for doc in valid_documents:
            for item in doc["evidence"]:

                label = item.get("label")

                if label in ["SUPPORT", "CONTRADICT"]:
                    original_label = label
                    break

            if original_label:
                break

        if original_label is None:
            stats["invalid_evidence"] += 1
            continue

        if original_label == "SUPPORT":
            stats["support"] += 1
        else:
            stats["contradict"] += 1

        output.append({
            "id": claim["id"],
            "source": "SciFact",
            "claim": claim["claim"],
            "original_label": original_label,
            "cited_documents": valid_documents
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in output:
            f.write(json.dumps(item) + "\n")

    print("\nSciFact processing complete")
    print("-" * 40)

    for key, value in stats.items():
        print(f"{key}: {value}")

    print("-" * 40)
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()