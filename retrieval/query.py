"""
Interactive CLI: type your own query, see TF-IDF and BM25 scores side by
side against the toy corpus, live.

Usage:
    python3 query.py
    > car oil change
    > automobile
    > quit
"""

from corpus import DOCUMENTS, DOC_LABELS
from tfidf import TfidfRetriever
from bm25 import Bm25Retriever


def main():
    tfidf = TfidfRetriever(DOCUMENTS)
    bm25 = Bm25Retriever(DOCUMENTS)

    print("Corpus:")
    for label in DOC_LABELS:
        print(f"  {label}")
    print("\nType a query and press enter (or 'quit' to exit).\n")

    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in {"quit", "exit"}:
            break

        tfidf_results = dict(tfidf.search(query, top_k=len(DOCUMENTS)))
        bm25_results = dict(bm25.search(query, top_k=len(DOCUMENTS)))
        ranked = sorted(bm25_results.items(), key=lambda kv: kv[1], reverse=True)

        print(f"  {'doc':<60} {'TF-IDF':>8} {'BM25':>8}")
        for doc_idx, bm25_score in ranked:
            if tfidf_results[doc_idx] == 0 and bm25_score == 0:
                continue  # skip non-matches for a cleaner live view
            print(f"  {DOC_LABELS[doc_idx]:<60} {tfidf_results[doc_idx]:>8.4f} {bm25_score:>8.4f}")
        print()


if __name__ == "__main__":
    main()
