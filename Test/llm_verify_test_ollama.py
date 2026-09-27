import json
import requests


# ============================================================
# CONFIG
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/chat"

# Change this when testing different models.
MODEL_NAME = "llama3.1:8b"


# ============================================================
# LLM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a scientific claim verification system.

Your task is to determine whether a CLAIM is supported,
contradicted, or not sufficiently supported by the supplied
evidence from a research paper.

You MUST use only the supplied evidence.
Do NOT use outside knowledge.
Do NOT assume facts that are not explicitly present
in the evidence.

Definitions:

SUPPORTED:
The evidence directly or clearly supports the claim.

CONTRADICTED:
The evidence directly or clearly contradicts the claim.

NOT ENOUGH EVIDENCE:
The evidence does not provide enough information to establish
the claim as either true or false.

Pay close attention to:
- numerical values
- percentages
- units
- experimental conditions
- material types
- comparisons
- qualifiers such as approximately, about, greater than, etc.

Return ONLY valid JSON in this format:

{
    "verdict": "SUPPORTED | CONTRADICTED | NOT ENOUGH EVIDENCE",
    "confidence": 0.0,
    "reason": "Short explanation based only on the supplied evidence.",
    "supporting_evidence": [1, 2],
    "contradicting_evidence": [3]
}

The evidence numbers refer to the numbered evidence sentences.
"""


# ============================================================
# CALL LOCAL LLM
# ============================================================

def call_llm(prompt: str) -> dict:
    """
    Send claim + retrieved evidence to Ollama.
    """

    payload = {
        "model": MODEL_NAME,

        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        "stream": False,

        "options": {
            "temperature": 0
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    content = data["message"]["content"].strip()

    # Remove markdown code fences if present
    if content.startswith("```"):
        content = content.replace("```json", "")
        content = content.replace("```", "")
        content = content.strip()

    return json.loads(content)


# ============================================================
# BUILD EVIDENCE PROMPT
# ============================================================

def build_prompt(
    claim: str,
    retrieved_evidence: list[tuple[int, str, float]]
) -> str:
    """
    Build the prompt using the SAME evidence retrieved
    by retrieve_test.py.

    Each item is:
        (sentence_index, sentence, relevance_score)
    """

    prompt = f"""
CLAIM:
{claim}

EVIDENCE FROM THE RESEARCH PAPER:

"""

    for i, (
        sentence_index,
        sentence,
        relevance
    ) in enumerate(
        retrieved_evidence,
        start=1
    ):

        prompt += f"""
--- EVIDENCE {i} ---
Sentence index: {sentence_index}
Retrieval relevance: {relevance:.3f}

{sentence}

"""

    prompt += """
Now determine whether the claim is SUPPORTED,
CONTRADICTED, or NOT ENOUGH EVIDENCE.

Return only the requested JSON.
"""

    return prompt


# ============================================================
# VERIFY CLAIM USING LLM
# ============================================================

def verify_claim_with_llm_test(
    claim: str,
    retrieved_evidence: list[tuple[int, str, float]]
) -> dict:
    """
    LLM-based benchmark verification.

    IMPORTANT:
    This function does NOT perform retrieval.

    It receives exactly the same retrieved evidence
    that is given to the NLI verifier.
    """

    # --------------------------------------------------------
    # 1. Build prompt from retrieved evidence
    # --------------------------------------------------------

    prompt = build_prompt(
        claim,
        retrieved_evidence
    )

    # --------------------------------------------------------
    # 2. Ask local LLM
    # --------------------------------------------------------

    llm_result = call_llm(prompt)

    # --------------------------------------------------------
    # 3. Return result
    # --------------------------------------------------------

    return {
        "claim": claim,

        "verdict": llm_result["verdict"],

        "confidence": float(
            llm_result["confidence"]
        ),

        "reason": llm_result["reason"],

        "supporting_evidence":
            llm_result.get(
                "supporting_evidence",
                []
            ),

        "contradicting_evidence":
            llm_result.get(
                "contradicting_evidence",
                []
            ),

        "evidence": retrieved_evidence
    }