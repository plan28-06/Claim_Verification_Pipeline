import pandas as pd
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent

FILES = [
    DATA_DIR / "train_set.csv",
    DATA_DIR / "validation_set.csv",
    DATA_DIR / "test_set.csv"
]

OUTPUT_FILE = DATA_DIR / "clinifact_pristine.jsonl"


def main():

    # Load all three splits
    dfs = [pd.read_csv(file) for file in FILES]
    df = pd.concat(dfs, ignore_index=True)

    output = []

    stats = {
        "total": len(df),
        "evidence": 0,
        "nei": 0,
        "inconclusive": 0
    }

    for _, row in df.iterrows():

        original_label = int(row["label"])

        # Ignore Inconclusive
        if original_label == 0:
            stats["inconclusive"] += 1
            continue

        # Evidence
        if original_label == 1:
            constructed_label = "SUPPORTED"
            stats["evidence"] += 1

        # NEI
        elif original_label == 2:
            constructed_label = "NOT ENOUGH EVIDENCE"
            stats["nei"] += 1

        else:
            continue

        record = {
            "id": int(row["index"]),
            "source": "CliniFact",
            "claim": str(row["claim"]),
            "pmid": str(row["PMID"]),
            "title": str(row["article_title"]),
            "abstract": str(row["article_abstract"]),
            "original_label": original_label,
            "label": constructed_label
        }

        output.append(record)

    # Save JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for record in output:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\nCliniFact processing complete")
    print("-" * 40)
    print(f"Total rows:       {stats['total']}")
    print(f"Evidence:         {stats['evidence']}")
    print(f"NEI:              {stats['nei']}")
    print(f"Inconclusive:     {stats['inconclusive']} (excluded)")
    print(f"Final records:    {len(output)}")
    print("-" * 40)
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()