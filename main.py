from ingest import ingest_pdf
from verify import verify_claim


# ---- Set these before running ----
PDF_PATH = r"D:\College\Internship\Papers\2006-Carturan-Strengthening of soda-lime silica glass by surface tereatement with sol-gel silica.pdf"
CLAIMS = [
    "Sol–gel silica coating increased the fracture resistance of annealed soda-lime-silica glass by approximately 35 MPa.",
    "Sol–gel silica coating increased the fracture resistance of ion-exchanged glass by approximately 35 MPa.",
    "The sol–gel coating process was performed using a spin-coating technique."

]



def print_result(result: dict):
    print(f"\nClaim: {result['claim']}")
    print(f"Verdict: {result['verdict']}")
    print("\nEvidence:")
    for item, score in zip(result["evidence"], result["retrieval_scores"]):
        print(f"- [{item['label']} (prob={item['probability']:.2f}), relevance={score:.2f}] {item['chunk'][:200]}...")
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