import json
import requests

from retrieve import retrieve_top_chunks


# ============================================================
# CONFIG
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/chat"

# Change this to whatever local model you have installed.
# Examples:
#   llama3.1:8b
#   mistral:7b
#   qwen2.5:7b
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
Do NOT assume facts that are not explicitly present in
the evidence.

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

The evidence numbers refer to the numbered evidence chunks.
"""


# ============================================================
# CALL LOCAL LLM
# ============================================================

def call_llm(prompt: str) -> dict:
    """
    Send the evidence and claim to the local Ollama model.
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

    content = data["message"]["content"]

    # Remove markdown code fences if the model adds them.
    content = content.strip()

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
    retrieved_chunks: list[tuple[str, float]]
) -> str:
    """
    Create the context that will be sent to the LLM.
    """

    prompt = f"""
CLAIM:
{claim}

EVIDENCE FROM THE RESEARCH PAPER:

"""

    for i, (chunk, relevance) in enumerate(
        retrieved_chunks,
        start=1
    ):

        prompt += f"""
--- EVIDENCE {i} ---
Retrieval relevance: {relevance:.3f}

{chunk}

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

def verify_claim_with_llm(
    claim: str,
    chunks: list[str],
    top_k: int = 5
) -> dict:
    """
    LLM-based claim verification.

    Pipeline:

        Claim
          ↓
        Retrieval
          ↓
        Top-K chunks
          ↓
        Local LLM
          ↓
        Verdict
    """

    # --------------------------------------------------------
    # 1. Retrieve the same type of evidence used by NLI
    # --------------------------------------------------------

    top_chunks = retrieve_top_chunks(
        claim,
        chunks,
        top_k=top_k
    )

    # --------------------------------------------------------
    # 2. Build LLM context
    # --------------------------------------------------------

    prompt = build_prompt(
        claim,
        top_chunks
    )

    # --------------------------------------------------------
    # 3. Ask local LLM to verify claim
    # --------------------------------------------------------

    llm_result = call_llm(prompt)

    # --------------------------------------------------------
    # 4. Return result
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

        "evidence": top_chunks,
    }



