import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


### Used solely for the retrieve top sentences function here

# ============================================================
# Frozen Retrieval Configuration
# ============================================================

SEMANTIC_WEIGHT = 0.9
BM25_WEIGHT = 0.1

# Load embedding model once
_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def _normalize(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize scores to [0, 1]."""

    min_val = scores.min()
    max_val = scores.max()

    if max_val - min_val < 1e-8:
        return np.zeros_like(scores)

    return (scores - min_val) / (max_val - min_val)


def semantic_scores(
    claim: str,
    sentences: list[str]
) -> np.ndarray:
    """Compute semantic similarity between claim and each sentence."""

    claim_vec = _embedding_model.encode(
        [claim],
        normalize_embeddings=True
    )[0]

    sentence_vecs = _embedding_model.encode(
        sentences,
        normalize_embeddings=True
    )

    return sentence_vecs @ claim_vec


def bm25_scores(
    claim: str,
    sentences: list[str]
) -> np.ndarray:
    """Compute BM25 lexical relevance scores."""

    tokenized_sentences = [
        sentence.lower().split()
        for sentence in sentences
    ]

    bm25 = BM25Okapi(tokenized_sentences)

    claim_tokens = claim.lower().split()

    return np.array(
        bm25.get_scores(claim_tokens)
    )


def retrieve_top_sentences(
    claim: str,
    sentences: list[str],
    top_k: int = 5,
) -> list[tuple[int, str, float]]:
    """
    Retrieve top-k sentences using the frozen
    0.9 semantic + 0.1 BM25 hybrid retriever.

    Returns:
        (sentence_index, sentence, relevance_score)
    """

    sem_scores = _normalize(
        semantic_scores(claim, sentences)
    )

    bm25_scores_array = _normalize(
        bm25_scores(claim, sentences)
    )

    relevance_scores = (
        SEMANTIC_WEIGHT * sem_scores
        + BM25_WEIGHT * bm25_scores_array
    )

    ranked_indices = np.argsort(
        relevance_scores
    )[::-1][:top_k]

    return [
        (
            int(idx),
            sentences[idx],
            float(relevance_scores[idx])
        )
        for idx in ranked_indices
    ]