"""
Side-by-side comparison of TF-IDF vs BM25 on the same toy corpus + queries.

Run directly to print a table per query. This is also the script used to
generate the example tables in the article writeup -- keep its output
format stable since the writeup quotes it.
"""

from corpus import DOCUMENTS, DOC_LABELS, QUERIES
from tfidf import TfidfRetriever
from bm25 import Bm25Retriever


def print_side_by_side(query: str, tfidf: TfidfRetriever, bm25: Bm25Retriever, top_k: int = 3):
    tfidf_results = dict(tfidf.search(query, top_k=len(DOCUMENTS)))
    bm25_results = dict(bm25.search(query, top_k=len(DOCUMENTS)))

    print(f'Query: "{query}"')
    print(f"  {'doc':<55} {'TF-IDF':>8} {'BM25':>8}")
    # order rows by BM25 rank for readability
    ranked = sorted(bm25_results.items(), key=lambda kv: kv[1], reverse=True)
    for doc_idx, bm25_score in ranked[:top_k]:
        tfidf_score = tfidf_results[doc_idx]
        print(f"  {DOC_LABELS[doc_idx]:<55} {tfidf_score:>8.4f} {bm25_score:>8.4f}")
    print()


if __name__ == "__main__":
    tfidf = TfidfRetriever(DOCUMENTS)
    bm25 = Bm25Retriever(DOCUMENTS)

    for query in QUERIES:
        print_side_by_side(query, tfidf, bm25)
