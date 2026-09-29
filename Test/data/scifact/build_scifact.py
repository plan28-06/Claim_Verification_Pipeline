import json
from pathlib import Path

DATA_DIR = Path(__file__).parent

TRAIN_FILE = DATA_DIR / "claims_train.jsonl"
DEV_FILE = DATA_DIR / "claims_dev.jsonl"
CORPUS_FILE = DATA_DIR / "corpus.jsonl"

TRAIN_OUTPUT = DATA_DIR / "scifact_train_pristine.jsonl"
DEV_OUTPUT = DATA_DIR / "scifact_dev_pristine.jsonl"


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def process_claims(claims, corpus_map):

    output = []

    stats = {
        "total": len(claims),
        "multi_document": 0,
        "support": 0,
        "contradict": 0,
        "no_evidence": 0,
        "invalid_source": 0,
        "invalid_evidence": 0
    }

    for claim in claims:

        evidence = claim.get("evidence", {})
        cited_doc_ids = claim.get("cited_doc_ids", [])

        # -----------------------------------------
        # Drop claims referencing multiple documents
        # -----------------------------------------

        if len(cited_doc_ids) != 1:
            stats["multi_document"] += 1
            continue

        if not evidence:
            stats["no_evidence"] += 1
            continue

        doc_id = cited_doc_ids[0]

        # -----------------------------------------
        # Find document in corpus
        # -----------------------------------------

        doc = corpus_map.get(str(doc_id))

        if doc is None:
            stats["invalid_source"] += 1
            continue

        doc_evidence = evidence.get(str(doc_id), [])

        if not doc_evidence:
            stats["invalid_source"] += 1
            continue

        abstract = doc.get("abstract", [])

        # -----------------------------------------
        # Validate evidence sentence indices
        # -----------------------------------------

        valid_evidence = []

        for item in doc_evidence:

            sentence_ids = item.get("sentences", [])

            if all(
                isinstance(idx, int)
                and 0 <= idx < len(abstract)
                for idx in sentence_ids
            ):
                valid_evidence.append(item)

        if not valid_evidence:
            stats["invalid_evidence"] += 1
            continue

        # -----------------------------------------
        # Determine label
        # -----------------------------------------

        original_label = None

        for item in valid_evidence:

            label = item.get("label")

            if label in ["SUPPORT", "CONTRADICT"]:
                original_label = label
                break

        if original_label is None:
            stats["invalid_evidence"] += 1
            continue

        if original_label == "SUPPORT":
            stats["support"] += 1
        else:
            stats["contradict"] += 1

        # -----------------------------------------
        # Create pristine record
        # -----------------------------------------

        output.append({
            "id": claim["id"],
            "source": "SciFact",
            "claim": claim["claim"],
            "original_label": original_label,
            "cited_documents": [
                {
                    "doc_id": doc_id,
                    "title": doc.get("title", ""),
                    "abstract": abstract,
                    "evidence": valid_evidence
                }
            ]
        })

    return output, stats


def save_jsonl(records, path):

    with open(path, "w", encoding="utf-8") as f:

        for item in records:

            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False
                ) + "\n"
            )


def print_stats(name, stats, output_path):

    print()
    print(name)
    print("-" * 50)

    for key, value in stats.items():
        print(f"{key}: {value}")

    print("-" * 50)
    print(f"Saved to: {output_path}")


def main():

    train = load_jsonl(TRAIN_FILE)
    dev = load_jsonl(DEV_FILE)
    corpus = load_jsonl(CORPUS_FILE)

    corpus_map = {
        str(doc["doc_id"]): doc
        for doc in corpus
    }

    # -----------------------------------------
    # TRAIN
    # -----------------------------------------

    train_output, train_stats = process_claims(
        train,
        corpus_map
    )

    save_jsonl(
        train_output,
        TRAIN_OUTPUT
    )

    print_stats(
        "SciFact TRAIN",
        train_stats,
        TRAIN_OUTPUT
    )

    # -----------------------------------------
    # DEV
    # -----------------------------------------

    dev_output, dev_stats = process_claims(
        dev,
        corpus_map
    )

    save_jsonl(
        dev_output,
        DEV_OUTPUT
    )

    print_stats(
        "SciFact DEV",
        dev_stats,
        DEV_OUTPUT
    )


if __name__ == "__main__":
    main()