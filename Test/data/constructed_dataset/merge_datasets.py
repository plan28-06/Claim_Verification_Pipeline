import json
from pathlib import Path

BASE_DIR = Path(__file__).parent

SCIFACT_FILE = BASE_DIR / "normalized_scifact.jsonl"
CLINIFACT_FILE = BASE_DIR / "normalized_clinifact.jsonl"
OUTPUT_FILE = BASE_DIR / "constructed_data.jsonl"


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():

    scifact = load_jsonl(SCIFACT_FILE)
    clinifact = load_jsonl(CLINIFACT_FILE)

    constructed = scifact + clinifact

    # Verify label distribution
    counts = {
        "SUPPORTED": 0,
        "CONTRADICTED": 0,
        "NOT ENOUGH EVIDENCE": 0
    }

    for record in constructed:
        label = record["label"]

        if label in counts:
            counts[label] += 1

    save_jsonl(constructed, OUTPUT_FILE)

    print("\nConstructed dataset created")
    print("-" * 50)
    print(f"SciFact records:              {len(scifact)}")
    print(f"CliniFact records:            {len(clinifact)}")
    print(f"Total records:                {len(constructed)}")
    print()
    print(f"SUPPORTED:                    {counts['SUPPORTED']}")
    print(f"CONTRADICTED:                 {counts['CONTRADICTED']}")
    print(f"NOT ENOUGH EVIDENCE:          {counts['NOT ENOUGH EVIDENCE']}")
    print("-" * 50)
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()