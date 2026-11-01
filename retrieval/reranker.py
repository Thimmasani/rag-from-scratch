"""
Cross-encoder reranking, Part 4 of the RAG-from-scratch series.

Bi-encoders (Part 2's embeddings.py) encode the query and each document
INDEPENDENTLY into fixed vectors, then compare those vectors with a dot
product. That independence is exactly what makes bi-encoders fast enough
to search a whole corpus: every document's embedding is precomputed once,
and a new query only needs one new encode + a matrix multiply.

A cross-encoder instead feeds the query and a document into the model
TOGETHER, as a single input, and outputs one relevance score directly --
no separate vectors, no dot product. This lets the model's attention
layers directly compare query tokens against document tokens, which is
more expressive than comparing two independently-computed summaries. The
cost: there's nothing to precompute. Every query-document pair needs a
full forward pass through the model, which is far too slow to run across
an entire corpus. That's why cross-encoders rerank a short candidate list
(typically 10-100 documents already retrieved by something fast) instead
of replacing retrieval.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2, trained on the MS MARCO
passage ranking dataset (real search queries paired with relevant/
irrelevant passages) -- a standard, widely used reranking model.
"""

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: list[str],
        candidate_indices: list[int],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """
        candidate_indices: doc indices already retrieved by something fast
        (BM25, embeddings, hybrid RRF) -- this does NOT search the full
        corpus, only re-scores this candidate list.
        """
        pairs = [(query, documents[idx]) for idx in candidate_indices]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(candidate_indices, scores), key=lambda pair: pair[1], reverse=True)
        ranked = [(idx, float(score)) for idx, score in ranked]
        return ranked[:top_k] if top_k else ranked


if __name__ == "__main__":
    from corpus import DOCUMENTS, DOC_LABELS, QUERIES
    from bm25 import Bm25Retriever
    from embeddings import EmbeddingRetriever
    from rrf import HybridRetriever

    bm25 = Bm25Retriever(DOCUMENTS)
    emb = EmbeddingRetriever(DOCUMENTS)
    hybrid = HybridRetriever([bm25, emb])
    reranker = CrossEncoderReranker()

    for query in QUERIES:
        candidates = [idx for idx, _score in hybrid.search(query, top_k=4)]
        print(f'Query: "{query}"')
        print(f"  hybrid candidates (RRF order): {candidates}")
        for idx, score in reranker.rerank(query, DOCUMENTS, candidates):
            print(f"  {score:>8.4f}  {DOC_LABELS[idx]}")
        print()
