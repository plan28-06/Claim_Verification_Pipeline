import json
import sys
from pathlib import Path


# ==================================================
# PATHS
# ==================================================

TEST_DIR = Path(__file__).resolve().parent

DATA_DIR = TEST_DIR / "data"
RESULTS_DIR = TEST_DIR / "results"

SCIFACT_DEV_FILE = (
    DATA_DIR
    / "scifact"
    / "scifact_dev_pristine.jsonl"
)


# ==================================================
# IMPORT RETRIEVER
# ==================================================

sys.path.insert(
    0,
    str(TEST_DIR)
)

from Test.retrieve_test import (
    retrieve_top_sentences
)


# ==================================================
# LOAD JSONL
# ==================================================

def load_jsonl(path):

    with open(path, "r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
        ]


# ==================================================
# RETRIEVAL METRICS
# ==================================================

def recall_at_k(
    results,
    gold_evidence,
    k
):

    retrieved_indices = {
        result[0]
        for result in results[:k]
    }

    gold_indices = set(
        gold_evidence
    )

    return int(
        bool(
            retrieved_indices
            & gold_indices
        )
    )


def reciprocal_rank(
    results,
    gold_evidence
):

    gold_indices = set(
        gold_evidence
    )

    for rank, result in enumerate(
        results,
        start=1
    ):

        sentence_index = result[0]

        if sentence_index in gold_indices:
            return 1.0 / rank

    return 0.0


# ==================================================
# EVALUATE ONE WEIGHT
# ==================================================

def evaluate_weight(
    records,
    semantic_weight
):

    recall_1 = []
    recall_3 = []
    recall_5 = []
    reciprocal_ranks = []

    for record in records:

        claim = record["claim"]

        # Each retained SciFact DEV record
        # has exactly one cited document.
        document = record[
            "cited_documents"
        ][0]

        sentences = document[
            "abstract"
        ]

        gold_evidence = []

        for evidence_item in document[
            "evidence"
        ]:

            for sentence_idx in evidence_item[
                "sentences"
            ]:

                if sentence_idx not in gold_evidence:
                    gold_evidence.append(
                        sentence_idx
                    )

        results = retrieve_top_sentences(
            claim=claim,
            sentences=sentences,
            top_k=5,
            semantic_weight=semantic_weight
        )

        recall_1.append(
            recall_at_k(
                results,
                gold_evidence,
                1
            )
        )

        recall_3.append(
            recall_at_k(
                results,
                gold_evidence,
                3
            )
        )

        recall_5.append(
            recall_at_k(
                results,
                gold_evidence,
                5
            )
        )

        reciprocal_ranks.append(
            reciprocal_rank(
                results,
                gold_evidence
            )
        )

    return {
        "recall@1":
            sum(recall_1) / len(recall_1),

        "recall@3":
            sum(recall_3) / len(recall_3),

        "recall@5":
            sum(recall_5) / len(recall_5),

        "mrr":
            sum(reciprocal_ranks)
            / len(reciprocal_ranks)
    }


# ==================================================
# SELECT BEST WEIGHT
# ==================================================

def select_weight(results):

    # Primary criterion:
    # highest MRR.
    #
    # Tie-breaker:
    # highest average of Recall@1,
    # Recall@3 and Recall@5.

    for result in results:

        result["mean_recall"] = (
            result["recall@1"]
            + result["recall@3"]
            + result["recall@5"]
        ) / 3

    best = max(
        results,
        key=lambda result: (
            result["mrr"],
            result["mean_recall"]
        )
    )

    return best


# ==================================================
# MAIN
# ==================================================

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    records = load_jsonl(
        SCIFACT_DEV_FILE
    )

    print()
    print("=" * 80)
    print("SciFact DEV Retrieval Weight Tuning")
    print("=" * 80)

    print(
        f"Dataset: {SCIFACT_DEV_FILE}"
    )

    print(
        f"Usable DEV records: {len(records)}"
    )

    print()

    # --------------------------------------------------
    # Sweep semantic weight from 0.0 to 1.0
    # --------------------------------------------------

    results = []

    for i in range(11):

        semantic_weight = round(
            i / 10,
            1
        )

        bm25_weight = round(
            1.0 - semantic_weight,
            1
        )

        print(
            f"Testing Semantic={semantic_weight:.1f}, "
            f"BM25={bm25_weight:.1f}..."
        )

        metrics = evaluate_weight(
            records,
            semantic_weight
        )

        results.append({

            "semantic_weight":
                semantic_weight,

            "bm25_weight":
                bm25_weight,

            **metrics
        })

    # --------------------------------------------------
    # Select weight
    # --------------------------------------------------

    best = select_weight(
        results
    )

    # --------------------------------------------------
    # Save complete sweep
    # --------------------------------------------------

    sweep_file = (
        RESULTS_DIR
        / "retrieval_weight_sweep.json"
    )

    with open(
        sweep_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    # --------------------------------------------------
    # Save selected weight
    # --------------------------------------------------

    selected_file = (
        RESULTS_DIR
        / "selected_retrieval_weight.json"
    )

    selection_info = {

        "semantic_weight":
            best["semantic_weight"],

        "bm25_weight":
            best["bm25_weight"],

        "dev_records":
            len(records),

        "selection_rule":
            "Highest MRR; average of Recall@1, "
            "Recall@3, and Recall@5 as tie-breaker.",

        "selected_recall@1":
            best["recall@1"],

        "selected_recall@3":
            best["recall@3"],

        "selected_recall@5":
            best["recall@5"],

        "selected_mrr":
            best["mrr"],

        "selected_mean_recall":
            best["mean_recall"]
    }

    with open(
        selected_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            selection_info,
            f,
            indent=2
        )

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(
        f"{'Semantic':>10} "
        f"{'BM25':>10} "
        f"{'R@1':>10} "
        f"{'R@3':>10} "
        f"{'R@5':>10} "
        f"{'MRR':>10} "
        f"{'Mean R':>10}"
    )

    print("-" * 80)

    for result in results:

        print(
            f"{result['semantic_weight']:>10.1f} "
            f"{result['bm25_weight']:>10.1f} "
            f"{result['recall@1']:>10.4f} "
            f"{result['recall@3']:>10.4f} "
            f"{result['recall@5']:>10.4f} "
            f"{result['mrr']:>10.4f} "
            f"{result['mean_recall']:>10.4f}"
        )

    # --------------------------------------------------
    # Selection
    # --------------------------------------------------

    print()
    print("=" * 80)
    print("SELECTION")
    print("=" * 80)

    print(
        "Primary criterion : Highest MRR"
    )

    print(
        "Tie-breaker       : "
        "Average of Recall@1, Recall@3 and Recall@5"
    )

    print()
    print(
        f"Selected semantic weight : "
        f"{best['semantic_weight']:.1f}"
    )

    print(
        f"Selected BM25 weight     : "
        f"{best['bm25_weight']:.1f}"
    )

    print(
        f"Recall@1                 : "
        f"{best['recall@1']:.4f}"
    )

    print(
        f"Recall@3                 : "
        f"{best['recall@3']:.4f}"
    )

    print(
        f"Recall@5                 : "
        f"{best['recall@5']:.4f}"
    )

    print(
        f"MRR                      : "
        f"{best['mrr']:.4f}"
    )

    print(
        f"Mean Recall              : "
        f"{best['mean_recall']:.4f}"
    )

    print()
    print(
        f"Sweep saved to: {sweep_file}"
    )

    print(
        f"Selected weight saved to: "
        f"{selected_file}"
    )


if __name__ == "__main__":
    main()
