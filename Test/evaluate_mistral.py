import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RESULTS_FILE = BASE_DIR / "results" / "mistral_results.json"
OUTPUT_FILE = BASE_DIR / "results" / "mistral_metrics.json"


# ============================================================
# LABELS
# ============================================================

LABELS = [
    "SUPPORTED",
    "CONTRADICTED",
    "NOT ENOUGH EVIDENCE",
]


# ============================================================
# LOAD RESULTS
# ============================================================

with open(RESULTS_FILE, "r", encoding="utf-8") as f:
    results = json.load(f)

print(f"Loaded {len(results)} records")


# ============================================================
# EXTRACT PREDICTIONS
# ============================================================

gold_labels = []
nli_predictions = []
mistral_predictions = []
hybrid_predictions = []

for item in results:

    gold_labels.append(item["gold_label"])

    nli_predictions.append(
        item["nli"]["verdict"]
    )

    mistral_predictions.append(
        item["mistral"]["verdict"]
    )

    hybrid_predictions.append(
        item["hybrid"]["mistral"]["verdict"]
    )


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_system(
    system_name,
    y_true,
    y_pred
):
    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0,
    )

    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS,
    )

    print("\n" + "=" * 70)
    print(system_name)
    print("=" * 70)

    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Macro Precision   : {macro_precision:.4f}")
    print(f"Macro Recall      : {macro_recall:.4f}")
    print(f"Macro F1          : {macro_f1:.4f}")

    # --------------------------------------------------------
    # Per-class metrics
    # --------------------------------------------------------

    print("\nPER-CLASS METRICS")
    print("-" * 100)

    print(
        f"{'Class':<25}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'Support':>12}"
    )

    print("-" * 100)

    per_class = {}

    for i, label in enumerate(LABELS):

        print(
            f"{label:<25}"
            f"{precision[i]:>12.4f}"
            f"{recall[i]:>12.4f}"
            f"{f1[i]:>12.4f}"
            f"{support[i]:>12}"
        )

        per_class[label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    print("\nCONFUSION MATRIX")
    print("-" * 60)

    print(
        f"{'Actual / Predicted':<25}"
        f"{'SUPPORTED':>15}"
        f"{'CONTRADICTED':>15}"
        f"{'NEI':>15}"
    )

    for i, label in enumerate(LABELS):

        print(
            f"{label:<25}"
            f"{cm[i][0]:>15}"
            f"{cm[i][1]:>15}"
            f"{cm[i][2]:>15}"
        )

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


# ============================================================
# EVALUATE SYSTEMS
# ============================================================

nli_metrics = evaluate_system(
    "DeBERTa NLI",
    gold_labels,
    nli_predictions,
)

mistral_metrics = evaluate_system(
    "Mistral 7B",
    gold_labels,
    mistral_predictions,
)

hybrid_metrics = evaluate_system(
    "NLI + Mistral 7B",
    gold_labels,
    hybrid_predictions,
)


# ============================================================
# HYBRID AGREEMENT
# ============================================================

agreement = sum(
    nli == mistral
    for nli, mistral
    in zip(nli_predictions, mistral_predictions)
)

disagreement = len(results) - agreement

agreement_percentage = agreement / len(results)
disagreement_percentage = disagreement / len(results)

print("\n" + "=" * 70)
print("HYBRID AGREEMENT")
print("=" * 70)

print(f"Total records      : {len(results)}")
print(
    f"Agreement          : {agreement} "
    f"({agreement_percentage:.4%})"
)
print(
    f"Disagreement       : {disagreement} "
    f"({disagreement_percentage:.4%})"
)


# ============================================================
# GOLD LABEL DISTRIBUTION
# ============================================================

gold_distribution = {}

for label in LABELS:
    count = gold_labels.count(label)

    gold_distribution[label] = {
        "count": count,
        "percentage": count / len(gold_labels),
    }

print("\n" + "=" * 70)
print("GOLD LABEL DISTRIBUTION")
print("=" * 70)

for label in LABELS:
    count = gold_distribution[label]["count"]
    percentage = gold_distribution[label]["percentage"]

    print(
        f"{label:<25}: "
        f"{count:>5} "
        f"({percentage:.2%})"
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    f"{'System':<25}"
    f"{'Accuracy':>12}"
    f"{'Macro-P':>12}"
    f"{'Macro-R':>12}"
    f"{'Macro-F1':>12}"
)

print("-" * 73)

summary_rows = [
    ("DeBERTa NLI", nli_metrics),
    ("Mistral 7B", mistral_metrics),
    ("NLI + Mistral", hybrid_metrics),
]

for name, metrics in summary_rows:

    print(
        f"{name:<25}"
        f"{metrics['accuracy']:>12.4f}"
        f"{metrics['macro_precision']:>12.4f}"
        f"{metrics['macro_recall']:>12.4f}"
        f"{metrics['macro_f1']:>12.4f}"
    )


# ============================================================
# SAVE METRICS
# ============================================================

output = {
    "num_records": len(results),

    "systems": {
        "deberta_nli": nli_metrics,
        "mistral_7b": mistral_metrics,
        "nli_plus_mistral": hybrid_metrics,
    },

    "hybrid_agreement": {
        "agreement": agreement,
        "disagreement": disagreement,
        "agreement_percentage": agreement_percentage,
        "disagreement_percentage": disagreement_percentage,
    },

    "gold_label_distribution": gold_distribution,
}


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4)


print("\n" + "=" * 70)
print(f"Metrics saved to: {OUTPUT_FILE}")
print("=" * 70)