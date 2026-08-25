from ingest import ingest_pdf
from verify import verify_claim


# ---- Set these before running ----
PDF_PATH = r".pdf"
CLAIMS = [
    "The alloy shows a yield strength of 450 MPa after annealing.",
]
# -----------------------------------


def print_result(result: dict):
    print(f"\nClaim: {result['claim']}")
    print(f"Verdict: {result['verdict']}")
    print("\nEvidence:")
    for item in result["evidence"]:
        print(f"- [{item['label']} ({item['confidence']:.2f})] {item['chunk'][:200]}...")
    print("\n" + "-" * 60)


def main():
    print(f"Ingesting {PDF_PATH}...")
    chunks = ingest_pdf(PDF_PATH)
    print(f"Done. {len(chunks)} chunks ready.")

    for claim in CLAIMS:
        result = verify_claim(claim, chunks, top_k=5)
        print_result(result)


if __name__ == "__main__":
    main()