import json
from pathlib import Path

from Test.llama.retrieve_test_ollama import retrieve_top_sentences
from Test.llama.verify_test_llama import verify_claim_test
from llm_verify_test_ollama import verify_claim_with_llm_test

# ============================================================
# CONFIG
# ============================================================

TOP_K = 5

# Temporary single example.
# Replace this with your SciFact/CliniFact loader later.
CLAIM = (
    "Sol–gel silica coating increased the fracture resistance "
    "of annealed soda-lime-silica glass by approximately 35 MPa."
)

ABSTRACT = """
Replace this with the abstract from SciFact or CliniFact.
"""


# ============================================================
# SPLIT ABSTRACT INTO SENTENCES
# ============================================================

def split_sentences(abstract: str) -> list[str]:
    """
    Split an abstract into individual sentences.
    """

    import re

    sentences = re.split(
        r'(?<=[.!?])\s+',
        abstract.strip()
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ============================================================
# COMBINE NLI + LLM
# ============================================================

def get_final_verdict(
    nli_verdict: str,
    llm_verdict: str
) -> str:

    if nli_verdict == llm_verdict:
        return nli_verdict

    return "NOT ENOUGH EVIDENCE"


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(
    claim: str,
    retrieved_evidence,
    nli_result,
    llm_result,
    final_verdict
):

    print("\n" + "=" * 70)

    print("\nCLAIM:")
    print(claim)

    print("\nRETRIEVED EVIDENCE:")

    for rank, (
        sentence_index,
        sentence,
        score
    ) in enumerate(
        retrieved_evidence,
        start=1
    ):

        print(
            f"\n[{rank}] "
            f"Sentence {sentence_index} "
            f"(Score: {score:.4f})"
        )

        print(sentence)

    print("\n" + "-" * 70)

    print(
        f"NLI Support Score: "
        f"{nli_result['support_score']:.4f}"
    )

    print(
        f"NLI Contradiction Score: "
        f"{nli_result['contradiction_score']:.4f}"
    )

    print(
        f"NLI Neutral Score: "
        f"{nli_result['neutral_score']:.4f}"
    )

    print(
        f"\nNLI Verdict: "
        f"{nli_result['verdict']}"
    )

    print(
        f"LLM Verdict: "
        f"{llm_result['verdict']}"
    )

    print(
        f"\nFINAL HYBRID VERDICT: "
        f"{final_verdict}"
    )

    print("\n" + "=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Split abstract into sentences
    # --------------------------------------------------------

    sentences = split_sentences(ABSTRACT)

    print(
        f"Abstract contains "
        f"{len(sentences)} sentences."
    )

    # --------------------------------------------------------
    # 2. Hybrid retrieval
    # --------------------------------------------------------

    retrieved_evidence = retrieve_top_sentences(
        CLAIM,
        sentences,
        top_k=TOP_K
    )

    # --------------------------------------------------------
    # 3. NLI verification
    # --------------------------------------------------------

    nli_result = verify_claim_test(
        CLAIM,
        retrieved_evidence
    )

    # --------------------------------------------------------
    # 4. LLM verification
    # --------------------------------------------------------

    llm_result = verify_claim_with_llm_test(
        CLAIM,
        retrieved_evidence
    )

    # --------------------------------------------------------
    # 5. Hybrid verdict
    # --------------------------------------------------------

    final_verdict = get_final_verdict(
        nli_result["verdict"],
        llm_result["verdict"]
    )

    # --------------------------------------------------------
    # 6. Print
    # --------------------------------------------------------

    print_result(
        CLAIM,
        retrieved_evidence,
        nli_result,
        llm_result,
        final_verdict
    )


if __name__ == "__main__":
    main()