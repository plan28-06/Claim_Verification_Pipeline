import json
import os
import sys
import tempfile
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

TEST_DIR = Path(__file__).resolve().parent

CACHE_FILE = TEST_DIR / "retrieval_cache.json"

RESULTS_DIR = TEST_DIR / "results"

OUTPUT_FILE = (
    RESULTS_DIR
    / "mistral_results.json"
)

# ============================================================
# IMPORT MODEL VERIFIERS
# ============================================================

sys.path.insert(
    0,
    str(TEST_DIR)
)

from nli_verify import verify_claim_test

# from llama.llm_verify_test_llama import (
#     verify_claim_with_llm as verify_llama
# )

# from qwen.llm_verify_test_qwen import (
#     verify_claim_with_llm as verify_qwen
# )

from mistral.llm_verify_test_mistral import (
    verify_claim_with_llm as verify_mistral
)


# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# FIRST RUN:
# Keep this at 5 to test the entire pipeline.
#
# FULL EXPERIMENT:
# Change to None after the 5-record test succeeds.
# ------------------------------------------------------------

LIMIT = None


# ============================================================
# LOAD RETRIEVAL CACHE
# ============================================================

def load_cache():

    if not CACHE_FILE.exists():

        raise FileNotFoundError(
            f"Retrieval cache not found:\n"
            f"{CACHE_FILE}"
        )

    with CACHE_FILE.open(
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# SAVE RESULTS
# ============================================================

def load_existing_results():
    """
    Load previously completed results so the experiment can resume
    after a restart/interruption.

    If no results file exists, start with an empty list.
    If the existing file is invalid JSON, stop instead of overwriting
    the previous checkpoint.
    """

    if not OUTPUT_FILE.exists():
        return []

    try:
        with OUTPUT_FILE.open(
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

    except json.JSONDecodeError as e:

        raise RuntimeError(
            f"Existing results file is corrupted and was not overwritten:\n"
            f"{OUTPUT_FILE}\n"
            f"JSON error: {e}"
        ) from e

    if not isinstance(data, list):

        raise RuntimeError(
            f"Existing results file does not contain a JSON list:\n"
            f"{OUTPUT_FILE}"
        )

    return data


def save_results(results):
    """
    Atomically save the checkpoint.

    The new JSON is first written to a temporary file in the same
    directory. Only after the write succeeds is it replaced over
    mistral_results.json.

    This prevents an interruption during writing from destroying
    the previous valid checkpoint.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=RESULTS_DIR,
            prefix="mistral_results_",
            suffix=".tmp",
            delete=False
        ) as f:

            temp_path = Path(f.name)

            json.dump(
                results,
                f,
                indent=2,
                ensure_ascii=False
            )

            f.flush()
            os.fsync(f.fileno())

        os.replace(
            temp_path,
            OUTPUT_FILE
        )

        temp_path = None

    finally:

        if temp_path is not None and temp_path.exists():

            try:
                temp_path.unlink()

            except OSError:
                pass


# ============================================================
# PREPARE NLI EVIDENCE
# ============================================================

def prepare_nli_evidence(
    retrieved_evidence
):
    """
    Convert retrieval cache format:

        {
            "sentence_index": ...,
            "sentence": ...,
            "relevance_score": ...
        }

    into the format expected by nli_verify.py:

        (sentence, relevance_score)
    """

    return [
        (
            item["sentence"],
            float(
                item["relevance_score"]
            )
        )

        for item in retrieved_evidence
    ]


# ============================================================
# HYBRID VERIFICATION
# ============================================================

def combine_hybrid(
    nli_verdict: str,
    llm_verdict: str
) -> dict:
    """
    Combine NLI and LLM predictions.

    If NLI and LLM agree:
        use the agreed verdict.

    If they disagree:
        return NOT ENOUGH EVIDENCE.
    """

    if nli_verdict == llm_verdict:

        verdict = nli_verdict

        agreement = True

    else:

        verdict = (
            "NOT ENOUGH EVIDENCE"
        )

        agreement = False

    return {
        "verdict": verdict,

        "agreement": agreement,

        "nli_verdict": nli_verdict,

        "llm_verdict": llm_verdict
    }


# ============================================================
# RUN CLASSIFICATION
# ============================================================

def main():

    # --------------------------------------------------------
    # Load frozen retrieval cache
    # --------------------------------------------------------

    cache = load_cache()

    total_records = len(cache)

    # --------------------------------------------------------
    # Apply test limit
    # --------------------------------------------------------

    if LIMIT is None:

        records = cache

    else:

        records = cache[:LIMIT]

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("SCIENTIFIC CLAIM CLASSIFICATION EXPERIMENT")
    print("=" * 75)

    print(
        f"Cached records : {total_records}"
    )

    print(
        f"Records to run : {len(records)}"
    )

    print()

    print(
        "Systems:"
    )

    print(
        "  1. DeBERTa NLI"
    )

    print(
        "  2. Llama 3.1 8B"
    )

    print(
        "  3. Qwen3 8B"
    )

    print(
        "  4. Mistral 7B"
    )

    print(
        "  5. DeBERTa + Llama"
    )

    print(
        "  6. DeBERTa + Qwen"
    )

    print(
        "  7. DeBERTa + Mistral"
    )

    print()

    # --------------------------------------------------------
    # Results / RESUME
    # --------------------------------------------------------

    results = load_existing_results()

    completed_ids = {
        item["id"]
        for item in results
        if isinstance(item, dict) and "id" in item
    }

    pending_records = [
        record
        for record in records
        if record["id"] not in completed_ids
    ]

    print(
        f"Previously completed: {len(completed_ids)}"
    )

    print(
        f"Remaining to run: {len(pending_records)}"
    )

    print()

    # --------------------------------------------------------
    # Process each claim
    # --------------------------------------------------------

    for i, record in enumerate(
        pending_records,
        start=1
    ):

        record_id = record["id"]

        source = record["source"]

        claim = record["claim"]

        gold_label = record["gold_label"]

        retrieved_evidence = (
            record["retrieved_evidence"]
        )

        print("=" * 75)

        print(
            f"[{len(completed_ids) + i}/{len(records)}] "
            f"{record_id}"
        )

        print(
            f"Source: {source}"
        )

        # ----------------------------------------------------
        # SAME RETRIEVED EVIDENCE FOR EVERY MODEL
        # ----------------------------------------------------

        nli_evidence = (
            prepare_nli_evidence(
                retrieved_evidence
            )
        )

        # ====================================================
        # 1. DeBERTa NLI
        # ====================================================

        print(
            "  [1/7] Running DeBERTa NLI..."
        )

        nli_result = verify_claim_test(
            claim,
            nli_evidence
        )

        print(
            f"        → {nli_result['verdict']}"
        )

        # ====================================================
        # 2. Llama 3.1 8B
        # ====================================================

        # print(
        #     "  [2/7] Running Llama 3.1 8B..."
        # )

        # llama_result = verify_llama(
        #     claim,
        #     retrieved_evidence
        # )

        # print(
        #     f"        → {llama_result['verdict']}"
        # )

        # ====================================================
        # 3. Qwen3 8B
        # ====================================================

        # print(
        #     "  [3/7] Running Qwen3 8B..."
        # )

        # qwen_result = verify_qwen(
        #     claim,
        #     retrieved_evidence
        # )

        # print(
        #     f"        → {qwen_result['verdict']}"
        # )

        # ====================================================
        # 4. Mistral 7B
        # ====================================================

        print(
            "  [4/7] Running Mistral 7B..."
        )

        mistral_result = verify_mistral(
            claim,
            retrieved_evidence
        )

        print(
            f"        → {mistral_result['verdict']}"
        )

        # ====================================================
        # 5. DeBERTa + Llama
        # ====================================================

        # print(
        #     "  [5/7] Computing DeBERTa + Llama hybrid..."
        # )

        # hybrid_llama = combine_hybrid(
        #     nli_result["verdict"],
        #     llama_result["verdict"]
        # )

        # print(
        #     f"        → {hybrid_llama['verdict']}"
        # )

        # ====================================================
        # 6. DeBERTa + Qwen
        # ====================================================

        # print(
        #     "  [6/7] Computing DeBERTa + Qwen hybrid..."
        # )

        # hybrid_qwen = combine_hybrid(
        #     nli_result["verdict"],
        #     qwen_result["verdict"]
        # )

        # print(
        #     f"        → {hybrid_qwen['verdict']}"
        # )

        # ====================================================
        # 7. DeBERTa + Mistral
        # ====================================================

        print(
            "  [7/7] Computing DeBERTa + Mistral hybrid..."
        )

        hybrid_mistral = combine_hybrid(
            nli_result["verdict"],
            mistral_result["verdict"]
        )

        print(
            f"        → {hybrid_mistral['verdict']}"
        )

        # ====================================================
        # SAVE RESULT
        # ====================================================

        result = {

            "id": record_id,

            "source": source,

            "claim": claim,

            "gold_label": gold_label,

            # Frozen evidence
            "retrieved_evidence":
                retrieved_evidence,

            # ------------------------------------------------
            # Individual classifiers
            # ------------------------------------------------

            "nli": nli_result,

            # "llama": llama_result,

            # "qwen": qwen_result,

            "mistral": mistral_result,

            # ------------------------------------------------
            # Hybrid systems
            # ------------------------------------------------

            "hybrid": {

                # "llama":
                #     hybrid_llama,

                # "qwen":
                #     hybrid_qwen,

                "mistral":
                    hybrid_mistral
            }
        }

        results.append(
            result
        )

        # ----------------------------------------------------
        # Checkpoint after every claim
        # ----------------------------------------------------

        save_results(
            results
        )

        # Mark this record completed only after the checkpoint
        # has been written successfully.
        completed_ids.add(record_id)

        print(
            f"        Checkpoint saved ({len(completed_ids)}/{len(records)})"
        )

        print()

    # ========================================================
    # COMPLETE
    # ========================================================

    print("=" * 75)

    print(
        "CLASSIFICATION EXPERIMENT COMPLETE"
    )

    print("=" * 75)

    print(
        f"Records available in results: {len(results)}"
    )

    print(
        f"Results saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()