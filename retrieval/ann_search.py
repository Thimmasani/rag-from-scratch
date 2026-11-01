"""
Brute-force vs. approximate nearest neighbor (HNSW) search, plus metadata
filtering, Part 8 of the RAG-from-scratch series.

BruteForceIndex and HnswIndex both operate on PRECOMPUTED embeddings, not
raw query strings -- unlike every other retriever in this series
(Bm25Retriever, EmbeddingRetriever, HybridRetriever, CrossEncoderReranker
all take a raw query and encode/tokenize it internally). That's a
deliberate, different signature here: the whole point of this part is to
isolate the search-structure question (brute-force sort vs. an ANN graph)
from the embedding step, so both indexes are built from the same
precomputed doc_embeddings array, and the caller (eval_ann.py) encodes
each query once via the one shared SentenceTransformer instance before
calling either index's .search().

Model: sentence-transformers/all-MiniLM-L6-v2 again (Part 2's model),
dim=384 -- HnswIndex's dim is hard-coded to match it, since this series
only ever uses this one embedding model.
"""

import numpy as np
import hnswlib


class BruteForceIndex:
    """Exact nearest-neighbor search: the accuracy baseline and the thing
    ANN is approximating. Same computation EmbeddingRetriever.search()
    already does (doc_embeddings @ query_embedding, full sort), isolated
    here as its own class so it can be benchmarked against HnswIndex on
    identical inputs."""

    def __init__(self, doc_embeddings: np.ndarray):
        self.doc_embeddings = doc_embeddings

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        scores = self.doc_embeddings @ query_embedding
        ranked = np.argsort(-scores)[:top_k]
        return [(int(idx), float(scores[idx])) for idx in ranked]


class HnswIndex:
    """Approximate nearest-neighbor search via hnswlib. ef_construction
    and M control graph build quality; ef_search controls query-time
    search width -- all three are left at hnswlib's own library defaults
    (ef_construction=200, M=16) except ef_search, which eval_ann.py sets
    explicitly and sweeps, since ef_search is the knob that trades recall
    for latency at query time without requiring a rebuild."""

    DIM = 384  # all-MiniLM-L6-v2's output dimension

    def __init__(self, doc_embeddings: np.ndarray, ef_search: int = 50):
        self.index = hnswlib.Index(space="cosine", dim=self.DIM)
        self.index.init_index(max_elements=len(doc_embeddings), ef_construction=200, M=16)
        self.index.add_items(doc_embeddings, np.arange(len(doc_embeddings)))
        self.index.set_ef(ef_search)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        labels, distances = self.index.knn_query(query_embedding, k=top_k)
        # hnswlib's "cosine" space returns a distance (1 - cosine_similarity);
        # convert back to similarity so results are directly comparable to
        # BruteForceIndex's scores.
        return [(int(label), float(1.0 - dist)) for label, dist in zip(labels[0], distances[0])]


def pre_filter_search(
    doc_embeddings: np.ndarray,
    metadata: list[dict],
    query_embedding: np.ndarray,
    filter_fn,
    top_k: int,
) -> list[tuple[int, float]]:
    """Filter by metadata FIRST, then search only within the matching
    subset. Guarantees up to top_k results as long as at least top_k
    documents match the filter at all."""
    matching_indices = [i for i, m in enumerate(metadata) if filter_fn(m)]
    if not matching_indices:
        return []
    sub_embeddings = doc_embeddings[matching_indices]
    scores = sub_embeddings @ query_embedding
    ranked_local = np.argsort(-scores)[:top_k]
    return [(matching_indices[i], float(scores[i])) for i in ranked_local]


def post_filter_search(
    index,
    query_embedding: np.ndarray,
    metadata: list[dict],
    filter_fn,
    top_k: int,
    candidate_k: int,
) -> list[tuple[int, float]]:
    """Search first (over candidate_k candidates from the index), THEN
    drop results failing filter_fn. Simpler -- no separate filtered-index
    machinery -- but can legitimately return fewer than top_k results, or
    zero, if a selective filter rejects most of the retrieved candidates."""
    candidates = index.search(query_embedding, candidate_k)
    filtered = [(idx, score) for idx, score in candidates if filter_fn(metadata[idx])]
    return filtered[:top_k]


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    from ann_corpus import generate_synthetic_corpus

    docs, meta = generate_synthetic_corpus(500, seed=42)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    doc_embeddings = model.encode(docs, normalize_embeddings=True)

    brute = BruteForceIndex(doc_embeddings)
    hnsw = HnswIndex(doc_embeddings)

    query_embedding = model.encode("durable wireless headphones", normalize_embeddings=True)
    print("Brute force top 3:", brute.search(query_embedding, 3))
    print("HNSW top 3:       ", hnsw.search(query_embedding, 3))
