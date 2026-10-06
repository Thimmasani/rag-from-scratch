"""
Hybrid retrieval via Reciprocal Rank Fusion (RRF), Part 3 of the
RAG-from-scratch series.

Problem: BM25 and embedding scores live on completely different, arbitrary
scales. BM25 scores in this corpus range from 0 to ~4.5 depending on term
rarity and query length; embedding cosine scores are bounded in [-1, 1].
Averaging or weighting these two numbers directly means picking an
arbitrary scale-matching constant that has no principled justification and
breaks the moment either retriever's score distribution shifts (e.g. a
longer query, a different embedding model, a bigger corpus).

RRF sidesteps the scale problem entirely by ignoring raw scores and fusing
based on RANK POSITION instead. Each retriever runs independently, each
produces a ranked list, and a document's fused score is just the sum of
1 / (k + rank) across every list it appears in. A low rank (near the top)
contributes a large reciprocal; a high rank (near the bottom) contributes
almost nothing. k is a smoothing constant (commonly 60) that controls how
much the fusion favors documents that are consistently ranked well across
retrievers vs documents that are #1 in one list but absent from the other.

    RRF(d) = sum over each retriever r of: 1 / (k + rank_r(d))

Documents missing entirely from one retriever's result list simply don't
get that term added -- no penalty beyond "doesn't contribute."
"""

from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """
    ranked_lists: one ranked list of doc indices per retriever, best first.
    Returns [(doc_idx, fused_score), ...] sorted by fused_score descending.
    """
    fused_scores: dict[int, float] = defaultdict(float)
    for ranked_list in ranked_lists:
        for rank, doc_idx in enumerate(ranked_list):  # rank is 0-indexed
            fused_scores[doc_idx] += 1 / (k + rank + 1)  # +1 -> rank 1 is top, not 0

    ranked = sorted(fused_scores.items(), key=lambda pair: pair[1], reverse=True)
    return ranked


class HybridRetriever:
    """Combines any number of retrievers that expose .search(query, top_k)."""

    def __init__(self, retrievers: list, k: int = 60):
        self.retrievers = retrievers
        self.k = k

    def search(self, query: str, top_k: int = 5, candidate_k: int = 10) -> list[tuple[int, float]]:
        ranked_lists = []
        for retriever in self.retrievers:
            results = retriever.search(query, top_k=candidate_k)
            ranked_lists.append([doc_idx for doc_idx, _score in results])

        fused = reciprocal_rank_fusion(ranked_lists, k=self.k)
        return fused[:top_k]


if __name__ == "__main__":
    from corpus import DOCUMENTS, DOC_LABELS, QUERIES
    from bm25 import Bm25Retriever
    from embeddings import EmbeddingRetriever

    bm25 = Bm25Retriever(DOCUMENTS)
    emb = EmbeddingRetriever(DOCUMENTS)
    hybrid = HybridRetriever([bm25, emb])

    for query in QUERIES:
        print(f'Query: "{query}"')
        for doc_idx, score in hybrid.search(query):
            print(f"  {score:.4f}  {DOC_LABELS[doc_idx]}")
        print()
