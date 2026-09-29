import json
from pathlib import Path

BASE_DIR = Path(__file__).parent

SCIFACT_FILE = BASE_DIR / "normalized_scifact.jsonl"
CLINIFACT_FILE = BASE_DIR / "normalized_clinifact.jsonl"

OUTPUT_FILE = BASE_DIR / "constructed_data.json"


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def main():
    scifact = load_jsonl(SCIFACT_FILE)
    clinifact = load_jsonl(CLINIFACT_FILE)

    constructed_data = scifact + clinifact

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            constructed_data,
            f,
            ensure_ascii=False,
            indent=2
        )

    # Count constructed labels
    counts = {}

    for record in constructed_data:
        label = record["label"]
        counts[label] = counts.get(label, 0) + 1

    print("\nConstructed dataset created")
    print("-" * 40)
    print(f"SciFact records:    {len(scifact)}")
    print(f"CliniFact records:  {len(clinifact)}")
    print(f"Total records:      {len(constructed_data)}")
    print("-" * 40)

    for label, count in counts.items():
        print(f"{label}: {count}")

    print("-" * 40)
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()