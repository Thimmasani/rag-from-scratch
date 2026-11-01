"""
Hybrid retrieval (Part 3) -> cross-encoder reranking (Part 4), side by side.
Source for the tables in writing/04_reranking.md -- keep output format
stable since the writeup quotes it.
"""

from corpus import (
    DOCUMENTS,
    DOC_LABELS,
    DOCUMENTS_WITH_INVOICE,
    DOC_LABELS_WITH_INVOICE,
    HYBRID_RESCUE_QUERY,
)
from bm25 import Bm25Retriever
from embeddings import EmbeddingRetriever
from rrf import HybridRetriever
from reranker import CrossEncoderReranker


def print_before_after(
    query: str,
    documents: list[str],
    doc_labels: list[str],
    hybrid: HybridRetriever,
    reranker: CrossEncoderReranker,
    candidate_k: int = 4,
):
    hybrid_results = hybrid.search(query, top_k=candidate_k)
    candidates = [idx for idx, _score in hybrid_results]
    reranked = reranker.rerank(query, documents, candidates)

    print(f'Query: "{query}"')
    print("  Hybrid (RRF) order:  ", [doc_labels[i].split(":")[0] for i in candidates])
    print("  Reranked order:      ", [doc_labels[i].split(":")[0] for i, _ in reranked])
    for idx, score in reranked:
        print(f"    {score:>9.4f}  {doc_labels[idx]}")
    print()


if __name__ == "__main__":
    bm25 = Bm25Retriever(DOCUMENTS)
    emb = EmbeddingRetriever(DOCUMENTS)
    hybrid = HybridRetriever([bm25, emb])
    reranker = CrossEncoderReranker()

    print("=== Case 1: reranker correctly breaks the RRF tie ===\n")
    print_before_after("weekend weather", DOCUMENTS, DOC_LABELS, hybrid, reranker)

    print("=== Case 2: reranker is fooled by exact-code overlap (same trap as embeddings) ===\n")
    print_before_after(HYBRID_RESCUE_QUERY, DOCUMENTS, DOC_LABELS, hybrid, reranker)

    print("=== Case 3: reranker is fooled by keyword stuffing (BM25 and embeddings were not) ===\n")
    print_before_after("oil change", DOCUMENTS, DOC_LABELS, hybrid, reranker)

    print("=== Case 4 (extended corpus): bare ID, reranker reproduces the wrong-document failure ===\n")
    bm25_ext = Bm25Retriever(DOCUMENTS_WITH_INVOICE)
    emb_ext = EmbeddingRetriever(DOCUMENTS_WITH_INVOICE)
    hybrid_ext = HybridRetriever([bm25_ext, emb_ext])
    print_before_after("INV-48213-B", DOCUMENTS_WITH_INVOICE, DOC_LABELS_WITH_INVOICE, hybrid_ext, reranker)
