"""
Index freshness: what changes when the corpus grows, Part 1 addition.

Uses the two corpora that already exist in corpus.py -- DOCUMENTS (6 docs)
and DOCUMENTS_WITH_INVOICE (the same 6 docs plus one new invoice doc,
already used by Part 2's wrong-document experiment). Neither is modified
here; this just runs both through BM25 and embeddings to measure, directly,
what "adding one document" actually costs each index type.

BM25's IDF for every term depends on N (total doc count) and df(t) (how
many docs contain that term) -- both are corpus-wide statistics. Adding a
single document can shift every other document's IDF values, even for
terms whose own document frequency didn't change, purely because N changed.
Embeddings have no such coupling: a document's embedding is a pure
function of that document's own text, so adding doc6 should leave doc0-5's
embeddings bit-for-bit identical.
"""

import numpy as np

from corpus import DOCUMENTS, DOCUMENTS_WITH_INVOICE
from bm25 import Bm25Retriever
from embeddings import EmbeddingRetriever


def bm25_idf_shift():
    small = Bm25Retriever(DOCUMENTS)
    large = Bm25Retriever(DOCUMENTS_WITH_INVOICE)

    print(f"N (small corpus) = {small.n_docs}, N (large corpus) = {large.n_docs}\n")
    print(f"{'term':<12} {'df (small)':<12} {'df (large)':<12} {'idf (small)':<14} {'idf (large)':<14} {'shifted?'}")

    shared_terms = sorted(set(small.idf) & set(large.idf))
    n_shifted = 0
    n_unchanged_df = 0
    for term in shared_terms:
        df_small = small.doc_freq[term]
        df_large = large.doc_freq[term]
        idf_small = small.idf[term]
        idf_large = large.idf[term]
        shifted = abs(idf_small - idf_large) > 1e-12
        if shifted:
            n_shifted += 1
        if df_small == df_large:
            n_unchanged_df += 1
            marker = "YES (df same!)" if shifted else "no"
        else:
            marker = "yes" if shifted else "no"
        print(f"{term:<12} {df_small:<12} {df_large:<12} {idf_small:<14.4f} {idf_large:<14.4f} {marker}")

    print(f"\n{n_shifted}/{len(shared_terms)} shared terms' IDF changed when doc6 was added.")
    print(f"Of those, {n_unchanged_df} terms shifted IDF despite their OWN df staying the same --")
    print("their weight moved purely because N (total doc count) changed underneath them.")


def embedding_stability():
    small = EmbeddingRetriever(DOCUMENTS)
    large = EmbeddingRetriever(DOCUMENTS_WITH_INVOICE)

    # First 6 rows of the large corpus's embeddings are docs 0-5, the same
    # documents small's embeddings cover.
    diff = np.abs(small.doc_embeddings - large.doc_embeddings[: len(DOCUMENTS)])
    max_diff = float(diff.max())

    print(f"\nmax absolute difference, doc0-5 embeddings, 6-doc corpus vs 7-doc corpus: {max_diff:.10f}")
    print("(bit-for-bit identical)" if max_diff == 0.0 else "(not identical)")


if __name__ == "__main__":
    print("=== BM25: IDF shift when one document is added ===\n")
    bm25_idf_shift()

    print("\n=== Embeddings: stability when one document is added ===")
    embedding_stability()
