import json
from pathlib import Path

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "results"
    / "dev_nli_results.json"
)

OUTPUT_FILE = (
    BASE_DIR
    / "results"
    / "nli_method_comparison.json"
)

TOP_K = 5

# Search thresholds from 0.50 to 0.90
THRESHOLDS = [
    round(x / 100, 2)
    for x in range(50, 91)
]

LABELS = [
    "SUPPORTED",
    "CONTRADICTED"
]


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("LOADING DEV NLI RESULTS")
print("=" * 70)

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    results = json.load(f)

print(
    f"Records: {len(results)}"
)

print(
    "Gold labels:"
)

support_count = sum(
    item["gold"] == "SUPPORTED"
    for item in results
)

contradict_count = sum(
    item["gold"] == "CONTRADICTED"
    for item in results
)

print(
    f"  SUPPORTED: {support_count}"
)

print(
    f"  CONTRADICTED: {contradict_count}"
)

print()


# ============================================================
# GOLD LABELS
# ============================================================

y_true = [
    item["gold"]
    for item in results
]


# ============================================================
# METHOD A
# CURRENT NON-NEUTRAL AGGREGATION
# ============================================================

def current_method(
    item,
    threshold
):

    chunks = sorted(
        item["evidence"],
        key=lambda c: c["retrieval_score"],
        reverse=True
    )[:TOP_K]

    support = 0.0
    contradiction = 0.0

    for chunk in chunks:

        relevance = chunk[
            "retrieval_score"
        ]

        probs = chunk[
            "all_probs"
        ]

        support += (
            relevance
            * probs["entailment"]
        )

        contradiction += (
            relevance
            * probs["contradiction"]
        )

    non_neutral = (
        support
        + contradiction
    )

    if non_neutral == 0:

        return "NOT ENOUGH EVIDENCE"

    support_ratio = (
        support
        / non_neutral
    )

    contradiction_ratio = (
        contradiction
        / non_neutral
    )

    if support_ratio >= threshold:

        return "SUPPORTED"

    elif contradiction_ratio >= threshold:

        return "CONTRADICTED"

    return "NOT ENOUGH EVIDENCE"


# ============================================================
# METHOD B
# MAX EVIDENCE
# ============================================================

def max_evidence_method(
    item,
    threshold
):

    chunks = sorted(
        item["evidence"],
        key=lambda c: c["retrieval_score"],
        reverse=True
    )[:TOP_K]

    if not chunks:

        return "NOT ENOUGH EVIDENCE"

    max_entailment = max(
        chunk["all_probs"]["entailment"]
        for chunk in chunks
    )

    max_contradiction = max(
        chunk["all_probs"]["contradiction"]
        for chunk in chunks
    )

    if (
        max_entailment >= threshold
        and max_entailment > max_contradiction
    ):

        return "SUPPORTED"

    elif (
        max_contradiction >= threshold
        and max_contradiction > max_entailment
    ):

        return "CONTRADICTED"

    return "NOT ENOUGH EVIDENCE"


# ============================================================
# METHOD C
# TOP-1 ONLY
# ============================================================

def top1_method(
    item,
    threshold
):

    chunks = sorted(
        item["evidence"],
        key=lambda c: c["retrieval_score"],
        reverse=True
    )

    if not chunks:

        return "NOT ENOUGH EVIDENCE"

    top_chunk = chunks[0]

    entailment = (
        top_chunk["all_probs"]["entailment"]
    )

    contradiction = (
        top_chunk["all_probs"]["contradiction"]
    )

    if (
        entailment >= threshold
        and entailment > contradiction
    ):

        return "SUPPORTED"

    elif (
        contradiction >= threshold
        and contradiction > entailment
    ):

        return "CONTRADICTED"

    return "NOT ENOUGH EVIDENCE"


# ============================================================
# EVALUATION
# ============================================================

def evaluate(predictions):

    # --------------------------------------------------------
    # Binary Macro-F1
    #
    # IMPORTANT:
    # NEI prediction is an error because DEV has no NEI gold.
    # --------------------------------------------------------

    macro_f1 = f1_score(
        y_true,
        predictions,
        labels=LABELS,
        average="macro",
        zero_division=0
    )

    macro_precision = precision_score(
        y_true,
        predictions,
        labels=LABELS,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_true,
        predictions,
        labels=LABELS,
        average="macro",
        zero_division=0
    )

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    return {
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "accuracy": accuracy
    }


# ============================================================
# SEARCH FUNCTION
# ============================================================

def search_method(
    method_name,
    method_function
):

    print("\n")
    print("=" * 70)
    print(f"METHOD: {method_name}")
    print("=" * 70)

    all_results = []

    for threshold in THRESHOLDS:

        predictions = [
            method_function(
                item,
                threshold
            )
            for item in results
        ]

        metrics = evaluate(
            predictions
        )

        result = {
            "threshold": threshold,
            **metrics
        }

        all_results.append(
            result
        )

        print(
            f"Threshold {threshold:.2f} | "
            f"F1 {metrics['macro_f1']:.4f} | "
            f"P {metrics['macro_precision']:.4f} | "
            f"R {metrics['macro_recall']:.4f} | "
            f"Acc {metrics['accuracy']:.4f}"
        )

    # --------------------------------------------------------
    # Select using Macro-F1
    # --------------------------------------------------------

    best = max(
        all_results,
        key=lambda x: (
            x["macro_f1"],
            x["macro_recall"],
            x["macro_precision"]
        )
    )

    print("\nBEST:")
    print(
        f"Threshold: {best['threshold']:.2f}"
    )

    print(
        f"Macro-F1: {best['macro_f1']:.4f}"
    )

    print(
        f"Precision: {best['macro_precision']:.4f}"
    )

    print(
        f"Recall: {best['macro_recall']:.4f}"
    )

    print(
        f"Accuracy: {best['accuracy']:.4f}"
    )

    return {
        "best": best,
        "all_thresholds": all_results
    }


# ============================================================
# RUN ALL THREE
# ============================================================

current_results = search_method(
    "CURRENT NON-NEUTRAL",
    current_method
)

max_results = search_method(
    "MAX EVIDENCE",
    max_evidence_method
)

top1_results = search_method(
    "TOP-1",
    top1_method
)


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n")
print("=" * 80)
print("FINAL FAIR DEV COMPARISON")
print("=" * 80)

print(
    f"{'METHOD':<25}"
    f"{'THRESHOLD':<12}"
    f"{'MACRO-F1':<12}"
    f"{'PRECISION':<12}"
    f"{'RECALL':<12}"
    f"{'ACCURACY':<10}"
)

print("-" * 80)

for name, data in [
    (
        "Current non-neutral",
        current_results
    ),
    (
        "Max evidence",
        max_results
    ),
    (
        "Top-1",
        top1_results
    )
]:

    best = data["best"]

    print(
        f"{name:<25}"
        f"{best['threshold']:<12.2f}"
        f"{best['macro_f1']:<12.4f}"
        f"{best['macro_precision']:<12.4f}"
        f"{best['macro_recall']:<12.4f}"
        f"{best['accuracy']:<10.4f}"
    )


# ============================================================
# SAVE
# ============================================================

comparison = {
    "dataset": {
        "records": len(results),
        "supported": support_count,
        "contradicted": contradict_count
    },

    "retrieval": {
        "semantic_weight": 0.9,
        "bm25_weight": 0.1,
        "top_k": TOP_K
    },

    "metric": {
        "primary": "binary_macro_f1",
        "secondary": [
            "macro_precision",
            "macro_recall",
            "accuracy"
        ]
    },

    "methods": {
        "current_non_neutral": current_results,
        "max_evidence": max_results,
        "top1": top1_results
    }
}

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        comparison,
        f,
        indent=2
    )


print("\n")
print("=" * 70)
print("COMPARISON SAVED")
print("=" * 70)

print(OUTPUT_FILE)

print("\nDone.")