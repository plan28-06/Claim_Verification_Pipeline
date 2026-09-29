import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from nli_verify import classify_chunk


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DEV_FILE = (
    BASE_DIR
    / "data"
    / "scifact"
    / "scifact_dev_pristine.jsonl"
)

OUTPUT_FILE = (
    BASE_DIR
    / "results"
    / "dev_nli_results.json"
)

SEMANTIC_WEIGHT = 0.9
BM25_WEIGHT = 0.1
TOP_K = 5


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.\n")


# ============================================================
# RETRIEVAL
# ============================================================

def normalize(scores):

    scores = np.asarray(scores)

    min_val = scores.min()
    max_val = scores.max()

    if max_val - min_val < 1e-8:
        return np.zeros_like(scores)

    return (
        (scores - min_val)
        / (max_val - min_val)
    )


def retrieve_top_sentences(
    claim,
    sentences,
    top_k=TOP_K
):

    # Semantic similarity
    claim_vec = embedding_model.encode(
        [claim],
        normalize_embeddings=True
    )[0]

    sentence_vecs = embedding_model.encode(
        sentences,
        normalize_embeddings=True
    )

    semantic_scores = sentence_vecs @ claim_vec

    # BM25
    tokenized_sentences = [
        sentence.lower().split()
        for sentence in sentences
    ]

    bm25 = BM25Okapi(
        tokenized_sentences
    )

    bm25_scores = np.array(
        bm25.get_scores(
            claim.lower().split()
        )
    )

    # Normalize exactly as production retriever
    semantic_scores = normalize(
        semantic_scores
    )

    bm25_scores = normalize(
        bm25_scores
    )

    # Frozen 0.9 / 0.1 hybrid
    relevance_scores = (
        SEMANTIC_WEIGHT * semantic_scores
        + BM25_WEIGHT * bm25_scores
    )

    ranked_indices = np.argsort(
        relevance_scores
    )[::-1][:top_k]

    return [
        {
            "sentence_index": int(idx),
            "sentence": sentences[idx],
            "retrieval_score": float(
                relevance_scores[idx]
            )
        }
        for idx in ranked_indices
    ]


# ============================================================
# LOAD DEV DATASET
# ============================================================

print("=" * 65)
print("LOADING SCIFACT DEV")
print("=" * 65)

records = []

with open(
    DEV_FILE,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        record = json.loads(line)

        # Only SUPPORT / CONTRADICT
        if record.get("original_label") not in {
            "SUPPORT",
            "CONTRADICT"
        }:
            continue

        # Only single-document claims
        if len(
            record.get("cited_documents", [])
        ) != 1:
            continue

        records.append(record)


print(
    f"Usable DEV records: {len(records)}"
)

print("=" * 65)


# ============================================================
# LABEL MAPPING
# ============================================================

def map_label(label):

    if label == "SUPPORT":
        return "SUPPORTED"

    if label == "CONTRADICT":
        return "CONTRADICTED"

    raise ValueError(
        f"Unknown label: {label}"
    )


# ============================================================
# RUN RETRIEVAL + NLI
# ============================================================

results = []

total = len(records)

print("\nStarting DEV processing...\n")


for i, record in enumerate(
    records,
    start=1
):

    claim = record["claim"]

    document = record[
        "cited_documents"
    ][0]

    sentences = document["abstract"]

    print(
        f"[{i:03d}/{total}] "
        f"{(i / total) * 100:6.2f}% | "
        f"ID: {record['id']} | "
        f"Gold: {record['original_label']}"
    )

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    retrieved = retrieve_top_sentences(
        claim,
        sentences
    )

    # --------------------------------------------------------
    # NLI
    # --------------------------------------------------------

    evidence = []

    for j, item in enumerate(
        retrieved,
        start=1
    ):

        print(
            f"    NLI {j}/{len(retrieved)}",
            end="\r",
            flush=True
        )

        nli = classify_chunk(
            item["sentence"],
            claim
        )

        evidence.append({
            "sentence_index":
                item["sentence_index"],

            "sentence":
                item["sentence"],

            "retrieval_score":
                item["retrieval_score"],

            "nli_label":
                nli["label"],

            "probability":
                nli["probability"],

            "all_probs":
                nli["all_probs"]
        })

    print(
        " " * 50,
        end="\r"
    )

    results.append({
        "id":
            record["id"],

        "claim":
            claim,

        "gold":
            map_label(
                record["original_label"]
            ),

        "evidence":
            evidence
    })


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )


print("\n")
print("=" * 65)
print("DEV NLI COMPLETE")
print("=" * 65)

print(
    f"Records processed: {len(results)}"
)

print(
    f"Top-K: {TOP_K}"
)

print(
    f"Semantic weight: {SEMANTIC_WEIGHT}"
)

print(
    f"BM25 weight: {BM25_WEIGHT}"
)

print(
    f"\nSaved to:\n{OUTPUT_FILE}"
)

print("=" * 65)