import json
import requests


# ============================================================
# CONFIG
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/chat"

MODEL_NAME = "mistral:7b"

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

Return ONLY valid JSON in exactly this format:

{
    "verdict": "SUPPORTED | CONTRADICTED | NOT ENOUGH EVIDENCE"
}
"""


# ============================================================
# CALL LOCAL LLM
# ============================================================

def call_llm(prompt: str) -> dict:
    """Send claim + retrieved evidence to Ollama."""

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

    # --------------------------------------------------------
    # Remove markdown code fences if the model adds them
    # --------------------------------------------------------

    if content.startswith("```"):
        content = content.replace("```json", "")
        content = content.replace("```", "")
        content = content.strip()

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    result = json.loads(content)

    # --------------------------------------------------------
    # Validate verdict
    # --------------------------------------------------------

    valid_verdicts = {
        "SUPPORTED",
        "CONTRADICTED",
        "NOT ENOUGH EVIDENCE"
    }

    verdict = result.get("verdict")

    if verdict not in valid_verdicts:
        raise ValueError(
            f"Invalid LLM verdict: {verdict}"
        )

    return {
        "verdict": verdict
    }


# ============================================================
# BUILD EVIDENCE PROMPT
# ============================================================

def build_prompt(
    claim: str,
    retrieved_evidence: list[dict]
) -> str:
    """
    Build the prompt from the frozen retrieval cache.

    Retrieval is NOT performed here.
    The function receives the exact evidence stored in
    retrieval_cache.json.
    """

    prompt = f"""
CLAIM:
{claim}

EVIDENCE FROM THE RESEARCH PAPER:

"""

    for i, evidence in enumerate(
        retrieved_evidence,
        start=1
    ):

        sentence_index = evidence["sentence_index"]
        sentence = evidence["sentence"]
        relevance = evidence["relevance_score"]

        prompt += f"""
--- EVIDENCE {i} ---
Sentence index: {sentence_index}
Retrieval relevance: {relevance:.3f}

{sentence}

"""

    prompt += """
Now determine whether the claim is SUPPORTED,
CONTRADICTED, or NOT ENOUGH EVIDENCE.

Return ONLY valid JSON in exactly this format:

{
    "verdict": "SUPPORTED | CONTRADICTED | NOT ENOUGH EVIDENCE"
}
"""

    return prompt


# ============================================================
# VERIFY CLAIM
# ============================================================

def verify_claim_with_llm(
    claim: str,
    retrieved_evidence: list[dict]
) -> dict:
    """
    LLM-based claim verification.

    IMPORTANT:
    This function does NOT perform retrieval.

    It receives the exact evidence stored in
    retrieval_cache.json.
    """

    prompt = build_prompt(
        claim,
        retrieved_evidence
    )

    llm_result = call_llm(prompt)

    return {
        "verdict": llm_result["verdict"]
    }