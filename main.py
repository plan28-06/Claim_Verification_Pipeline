from ingest import ingest_pdf
from verify import verify_claim
from llm_verify import verify_claim_with_llm


# ------------------------------------------------------------
# Set these before running
# ------------------------------------------------------------

PDF_PATH = (
    r"D:\College\Internship\Papers"
    r"\2006-Carturan-Strengthening of soda-lime silica glass "
    r"by surface tereatement with sol-gel silica.pdf"
)

CLAIMS = [

    "Sol–gel silica coating increased the fracture resistance "
    "of annealed soda-lime-silica glass by approximately 35 MPa.",

    "Sol–gel silica coating increased the fracture resistance "
    "of ion-exchanged glass by approximately 35 MPa.",

    "The sol–gel coating process was performed using "
    "a spin-coating technique."

]


# ------------------------------------------------------------
# Combine NLI + LLM verdicts
# ------------------------------------------------------------

def get_final_verdict(
    nli_verdict: str,
    llm_verdict: str
) -> str:
    """
    Final verdict is only considered reliable when
    NLI and LLM independently agree.

    If they disagree, return NOT ENOUGH EVIDENCE.
    """

    if nli_verdict == llm_verdict:

        if nli_verdict == "SUPPORTED":
            return "SUPPORTED"

        if nli_verdict == "CONTRADICTED":
            return "CONTRADICTED"

        if nli_verdict == "NOT ENOUGH EVIDENCE":
            return "NOT ENOUGH EVIDENCE"

    return "NOT ENOUGH EVIDENCE"


# ------------------------------------------------------------
# Print result
# ------------------------------------------------------------

def print_result(result: dict, llm_result: dict, final_verdict: str):

    print(f"\nClaim: {result['claim']}")

    # Highest individual NLI class + probability + relevance
    best_item = max(
        result["evidence"],
        key=lambda item: item["probability"]
    )

    print(
        f"NLI Support Score: {result['support_score']:.4f}\n"
        f"NLI Contradiction Score: {result['contradiction_score']:.4f}\n"
        f"NLI Neutral Score: {result['neutral_score']:.4f}\n"
    )

    # This is the aggregated NLI verdict
    print(
        f"NLI Verdict: {result['verdict']}"
    )

    print(
        f"LLM Verdict: {llm_result['verdict']}"
    )

    print(
        f"Final Combined Verdict: {final_verdict}"
    )

    print("\n" + "-" * 60)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print(f"Ingesting {PDF_PATH}...")

    chunks = ingest_pdf(PDF_PATH)

    print(f"Done. {len(chunks)} chunks ready.")

    for claim in CLAIMS:

        # NLI verification
        result = verify_claim(
            claim,
            chunks,
            top_k=5
        )

        # LLM verification
        llm_result = verify_claim_with_llm(
            claim,
            chunks,
            top_k=5
        )

        # Combine NLI + LLM
        if result["verdict"] == llm_result["verdict"]:
            final_verdict = result["verdict"]
        else:
            final_verdict = "NOT ENOUGH EVIDENCE"

        print_result(
            result,
            llm_result,
            final_verdict
        )


if __name__ == "__main__":
    main()