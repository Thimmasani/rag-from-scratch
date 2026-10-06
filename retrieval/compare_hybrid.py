"""
Side-by-side comparison of BM25, embeddings, and RRF-fused hybrid retrieval.
Source for the tables in writing/03_hybrid_rrf.md -- keep output format
stable since the writeup quotes it.
"""

from corpus import DOCUMENTS, DOC_LABELS, QUERIES, HYBRID_RESCUE_QUERY
from bm25 import Bm25Retriever
from embeddings import EmbeddingRetriever
from rrf import HybridRetriever


def print_comparison(query: str, bm25: Bm25Retriever, emb: EmbeddingRetriever, hybrid: HybridRetriever, top_k: int = 3):
    bm25_results = dict(bm25.search(query, top_k=len(DOCUMENTS)))
    emb_results = dict(emb.search(query, top_k=len(DOCUMENTS)))
    hybrid_results = hybrid.search(query, top_k=top_k)

    print(f'Query: "{query}"')
    print(f"  {'doc':<55} {'BM25':>8} {'Embed':>8} {'RRF':>10}")
    for doc_idx, rrf_score in hybrid_results:
        print(f"  {DOC_LABELS[doc_idx]:<55} {bm25_results[doc_idx]:>8.4f} {emb_results[doc_idx]:>8.4f} {rrf_score:>10.6f}")
    print()


if __name__ == "__main__":
    bm25 = Bm25Retriever(DOCUMENTS)
    emb = EmbeddingRetriever(DOCUMENTS)
    hybrid = HybridRetriever([bm25, emb])

    for query in QUERIES:
        print_comparison(query, bm25, emb, hybrid)

    print("=== The RRF tie case: embeddings alone are wrong, hybrid pulls it to a tie ===\n")
    print_comparison(HYBRID_RESCUE_QUERY, bm25, emb, hybrid)
