"""
Side-by-side comparison of TF-IDF, BM25, and embeddings on the same toy
corpus + queries. Source for the tables in writing/02_embeddings.md --
keep output format stable since the writeup quotes it.
"""

from corpus import DOCUMENTS, DOC_LABELS, QUERIES, UNSEEN_CODE_QUERIES
from tfidf import TfidfRetriever
from bm25 import Bm25Retriever
from embeddings import EmbeddingRetriever


def print_three_way(query: str, tfidf: TfidfRetriever, bm25: Bm25Retriever, emb: EmbeddingRetriever, top_k: int = 4):
    tfidf_results = dict(tfidf.search(query, top_k=len(DOCUMENTS)))
    bm25_results = dict(bm25.search(query, top_k=len(DOCUMENTS)))
    emb_results = dict(emb.search(query, top_k=len(DOCUMENTS)))

    print(f'Query: "{query}"')
    print(f"  {'doc':<55} {'TF-IDF':>8} {'BM25':>8} {'Embed':>8}")
    ranked = sorted(emb_results.items(), key=lambda kv: kv[1], reverse=True)
    for doc_idx, emb_score in ranked[:top_k]:
        print(f"  {DOC_LABELS[doc_idx]:<55} {tfidf_results[doc_idx]:>8.4f} {bm25_results[doc_idx]:>8.4f} {emb_score:>8.4f}")
    print()


def print_unseen_code_experiment(emb: EmbeddingRetriever):
    print("=== Unseen code experiment (embeddings only) ===\n")
    for query in UNSEEN_CODE_QUERIES:
        print(f'Query: "{query}"')
        for doc_idx, score in emb.search(query, top_k=3):
            print(f"  {score:.4f}  {DOC_LABELS[doc_idx]}")
        print()


if __name__ == "__main__":
    tfidf = TfidfRetriever(DOCUMENTS)
    bm25 = Bm25Retriever(DOCUMENTS)
    emb = EmbeddingRetriever(DOCUMENTS)

    for query in QUERIES:
        print_three_way(query, tfidf, bm25, emb)

    print_unseen_code_experiment(emb)
