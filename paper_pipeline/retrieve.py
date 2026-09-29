import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from ingest import ingest_pdf  

# Load once — reused across calls
_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def _normalize(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize a score array to 0-1 range."""
    min_val, max_val = scores.min(), scores.max()
    if max_val - min_val < 1e-8:
        return np.zeros_like(scores)  # avoid divide-by-zero if all scores equal
    return (scores - min_val) / (max_val - min_val)


def semantic_scores(claim: str, chunks: list[str]) -> np.ndarray:
    """Step 1: Cosine similarity between claim and each chunk."""
    claim_vec = _embedding_model.encode([claim])[0]
    chunk_vecs = _embedding_model.encode(chunks)

    # Cosine similarity = dot product / (norm * norm)
    claim_norm = claim_vec / np.linalg.norm(claim_vec)
    chunk_norms = chunk_vecs / np.linalg.norm(chunk_vecs, axis=1, keepdims=True)
    sims = chunk_norms @ claim_norm  # dot product of each chunk with the claim

    return sims


def bm25_scores(claim: str, chunks: list[str]) -> np.ndarray:
    """Step 2: BM25 lexical relevance score for each chunk."""
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    tokenized_claim = claim.lower().split()
    scores = bm25.get_scores(tokenized_claim)

    return np.array(scores)


def retrieve_top_chunks(
    claim: str,
    chunks: list[str],
    top_k: int = 5,
    semantic_weight: float = 0.5,
) -> list[tuple[str, float]]:
    """Full Stage 2 pipeline: claim + chunks -> top-k (chunk, score) pairs."""

    sem_scores = _normalize(semantic_scores(claim, chunks))
    lex_scores = _normalize(bm25_scores(claim, chunks))

    relevance_score = semantic_weight * sem_scores + (1 - semantic_weight) * lex_scores

    # Pair chunks with scores, sort descending, take top-k
    ranked = sorted(zip(chunks, relevance_score), key=lambda pair: pair[1], reverse=True)
    return ranked[:top_k]


