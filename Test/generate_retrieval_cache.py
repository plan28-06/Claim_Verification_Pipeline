import json
import sys
from pathlib import Path


# ============================================================
# Paths
# ============================================================

TEST_DIR = Path(__file__).resolve().parent

DATA_FILE = (
    TEST_DIR
    / "data"
    / "constructed_dataset"
    / "constructed_data.jsonl"
)

OUTPUT_FILE = TEST_DIR / "retrieval_cache.json"


# ============================================================
# Import frozen retriever
# ============================================================

sys.path.insert(0, str(TEST_DIR))

from Test.retrieve_test import retrieve_top_sentences


# ============================================================
# Evidence extraction
# ============================================================

def get_source_sentences(record: dict) -> list[str]:
    """
    Extract the source sentences from the normalized
    constructed dataset.

    Both SciFact and CliniFact use:
        record["abstract"]
    """

    sentences = record.get("abstract", [])

    if not sentences:
        raise ValueError(
            f"No source sentences found for record {record.get('id')}"
        )

    return sentences

# ============================================================
# Main
# ============================================================

def main():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_FILE}"
        )

    print("=" * 60)
    print("Generating Retrieval Cache")
    print("=" * 60)

    print(f"Input : {DATA_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print()

    cache = []

    with DATA_FILE.open("r", encoding="utf-8") as f:

        records = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    total = len(records)

    print(f"Total records: {total}")
    print()

    for i, record in enumerate(records, start=1):

        record_id = record["id"]
        source = record["source"]
        claim = record["claim"]
        gold_label = record["label"]

        sentences = get_source_sentences(record)

        if not sentences:
            raise ValueError(
                f"No source sentences found for record {record_id}"
            )

        # ----------------------------------------------------
        # IMPORTANT:
        # Retrieval is performed EXACTLY ONCE per claim.
        # ----------------------------------------------------

        retrieved = retrieve_top_sentences(
            claim=claim,
            sentences=sentences,
            top_k=5
        )

        retrieved_evidence = []

        for sentence_index, sentence, relevance_score in retrieved:

            retrieved_evidence.append(
                {
                    "sentence_index": sentence_index,
                    "sentence": sentence,
                    "relevance_score": relevance_score
                }
            )

        cache.append(
            {
                "id": record_id,
                "source": source,
                "claim": claim,
                "gold_label": gold_label,
                "retrieved_evidence": retrieved_evidence
            }
        )

        if i % 100 == 0 or i == total:
            print(f"Processed {i}/{total}")

    # ========================================================
    # Save cache
    # ========================================================

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:

        json.dump(
            cache,
            f,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # Validation
    # ========================================================

    print()
    print("=" * 60)
    print("Retrieval cache generated successfully")
    print("=" * 60)

    print(f"Records written : {len(cache)}")
    print(f"Output file     : {OUTPUT_FILE}")

    if len(cache) != total:
        raise RuntimeError(
            f"Cache size mismatch: "
            f"{len(cache)} != {total}"
        )

    missing_evidence = sum(
        1
        for item in cache
        if not item["retrieved_evidence"]
    )

    print(f"Missing evidence: {missing_evidence}")

    if missing_evidence > 0:
        raise RuntimeError(
            "Some records have no retrieved evidence."
        )

    print()
    print("Retrieval configuration:")
    print("  Semantic weight = 0.9")
    print("  BM25 weight     = 0.1")
    print("  Top-k            = 5")
    print()
    print("Retrieval is now cached and should NOT be rerun")
    print("during the classifier comparison experiments.")


if __name__ == "__main__":
    main()