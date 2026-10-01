import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RESULTS_FILE = BASE_DIR / "results" / "llama_results.json"

LABELS = [
    "SUPPORTED",
    "CONTRADICTED",
    "NOT ENOUGH EVIDENCE",
]


# ============================================================
# LOAD RESULTS
# ============================================================

with open(RESULTS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} records")


# ============================================================
# EXTRACT PREDICTIONS
# ============================================================

gold = []
nli_predictions = []
llama_predictions = []
hybrid_predictions = []

for record in data:

    gold.append(record["gold_label"])

    # DeBERTa NLI
    nli_predictions.append(
        record["nli"]["verdict"]
    )

    # Llama
    llama_predictions.append(
        record["llama"]["verdict"]
    )

    # NLI + Llama hybrid
    hybrid_predictions.append(
        record["hybrid"]["llama"]["verdict"]
    )


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_system(name, y_true, y_pred):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=LABELS,
            zero_division=0
        )
    )

    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()

    # ========================================================
    # OVERALL
    # ========================================================

    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Macro Precision   : {macro_precision:.4f}")
    print(f"Macro Recall      : {macro_recall:.4f}")
    print(f"Macro F1          : {macro_f1:.4f}")

    # ========================================================
    # PER-CLASS
    # ========================================================

    print()
    print("PER-CLASS METRICS")
    print("-" * 80)

    print(
        f"{'Class':25s}"
        f"{'Precision':>12s}"
        f"{'Recall':>12s}"
        f"{'F1':>12s}"
        f"{'Support':>12s}"
    )

    print("-" * 80)

    for label, p, r, f, s in zip(
        LABELS,
        precision,
        recall,
        f1,
        support
    ):

        print(
            f"{label:25s}"
            f"{p:12.4f}"
            f"{r:12.4f}"
            f"{f:12.4f}"
            f"{s:12d}"
        )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS
    )

    print()
    print("CONFUSION MATRIX")
    print("-" * 80)

    print(
        f"{'Actual \\ Predicted':25s}"
        f"{'SUPPORTED':>18s}"
        f"{'CONTRADICTED':>18s}"
        f"{'NEI':>18s}"
    )

    for label, row in zip(LABELS, cm):

        print(
            f"{label:25s}"
            f"{row[0]:18d}"
            f"{row[1]:18d}"
            f"{row[2]:18d}"
        )

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),

        "per_class": {
            label: {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(s),
            }
            for label, p, r, f, s in zip(
                LABELS,
                precision,
                recall,
                f1,
                support
            )
        },

        "confusion_matrix": cm.tolist(),
    }


# ============================================================
# EVALUATE ALL THREE SYSTEMS
# ============================================================

nli_results = evaluate_system(
    "DeBERTa NLI",
    gold,
    nli_predictions
)

llama_results = evaluate_system(
    "Llama 3.1 8B",
    gold,
    llama_predictions
)

hybrid_results = evaluate_system(
    "NLI + Llama 3.1 8B",
    gold,
    hybrid_predictions
)


# ============================================================
# HYBRID AGREEMENT
# ============================================================

agreement_count = 0
disagreement_count = 0

for record in data:

    nli = record["hybrid"]["llama"]["nli_verdict"]
    llama = record["hybrid"]["llama"]["llm_verdict"]

    if nli == llama:
        agreement_count += 1
    else:
        disagreement_count += 1


total = len(data)

print()
print("=" * 80)
print("HYBRID AGREEMENT")
print("=" * 80)

print(f"Total records       : {total}")

print(
    f"Agreement           : {agreement_count} "
    f"({agreement_count / total:.4%})"
)

print(
    f"Disagreement        : {disagreement_count} "
    f"({disagreement_count / total:.4%})"
)


# ============================================================
# GOLD LABEL DISTRIBUTION
# ============================================================

print()
print("=" * 80)
print("GOLD LABEL DISTRIBUTION")
print("=" * 80)

for label in LABELS:

    count = gold.count(label)

    print(
        f"{label:25s}: "
        f"{count:5d} "
        f"({count / total:.2%})"
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

print(
    f"{'System':25s}"
    f"{'Accuracy':>12s}"
    f"{'Macro-P':>12s}"
    f"{'Macro-R':>12s}"
    f"{'Macro-F1':>12s}"
)

print("-" * 73)

for name, result in [
    ("DeBERTa NLI", nli_results),
    ("Llama 3.1 8B", llama_results),
    ("NLI + Llama", hybrid_results),
]:

    print(
        f"{name:25s}"
        f"{result['accuracy']:12.4f}"
        f"{result['macro_precision']:12.4f}"
        f"{result['macro_recall']:12.4f}"
        f"{result['macro_f1']:12.4f}"
    )


# ============================================================
# SAVE METRICS
# ============================================================

output = {
    "num_records": total,

    "class_distribution": {
        label: gold.count(label)
        for label in LABELS
    },

    "nli": nli_results,
    "llama": llama_results,
    "hybrid_llama": hybrid_results,

    "hybrid_agreement": {
        "agreement": agreement_count,
        "disagreement": disagreement_count,
        "agreement_rate": agreement_count / total,
        "disagreement_rate": disagreement_count / total,
    },
}

output_file = BASE_DIR / "results" / "llama_metrics.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(
        output,
        f,
        indent=4
    )

print()
print(f"Metrics saved to: {output_file}")