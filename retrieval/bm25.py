"""
BM25 from scratch, no rank_bm25 / sklearn.

BM25 (Okapi BM25) fixes two weaknesses of raw TF-IDF:
  1. TF saturation: a term appearing 10x shouldn't score ~10x higher than
     appearing once. Diminishing returns via the `k1` parameter.
  2. Document length normalization: a term match in a short doc is stronger
     evidence of relevance than the same match in a long doc stuffed with
     many terms. Controlled by `b`.

Formula, per term t in query, per document d:

    score(d, q) = sum over t in q of:
        IDF(t) * ( f(t, d) * (k1 + 1) )
                  -----------------------------------------
                  ( f(t, d) + k1 * (1 - b + b * |d| / avgdl) )

    IDF(t) = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )
        N      = total number of documents
        df(t)  = number of documents containing term t
        f(t,d) = count of term t in document d
        |d|    = length of document d (token count)
        avgdl  = average document length across the corpus

    k1 (typically 1.2-2.0): controls TF saturation speed.
    b  (typically 0.75): controls strength of length normalization.
"""

import math
from collections import Counter

from tfidf import tokenize  # reuse the same tokenizer for a fair comparison


class Bm25Retriever:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        self.documents = documents
        self.tokenized_docs = [tokenize(doc) for doc in documents]
        self.n_docs = len(documents)
        self.doc_lengths = [len(tokens) for tokens in self.tokenized_docs]
        self.avg_doc_length = sum(self.doc_lengths) / self.n_docs

        self.doc_term_freqs = [Counter(tokens) for tokens in self.tokenized_docs]

        self.doc_freq = self._compute_doc_freq()
        self.idf = self._compute_idf()

    def _compute_doc_freq(self) -> Counter:
        df = Counter()
        for tokens in self.tokenized_docs:
            for term in set(tokens):
                df[term] += 1
        return df

    def _compute_idf(self) -> dict[str, float]:
        idf = {}
        for term, df in self.doc_freq.items():
            idf[term] = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
        return idf

    def _score(self, query_tokens: list[str], doc_idx: int) -> float:
        score = 0.0
        doc_len = self.doc_lengths[doc_idx]
        term_freqs = self.doc_term_freqs[doc_idx]
        length_norm = 1 - self.b + self.b * (doc_len / self.avg_doc_length)

        for term in query_tokens:
            if term not in self.idf:
                continue  # unseen term contributes nothing, same as TF-IDF
            f = term_freqs.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * length_norm
            score += self.idf[term] * (numerator / denominator)

        return score

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        query_tokens = tokenize(query)
        scores = [
            (i, self._score(query_tokens, i)) for i in range(self.n_docs)
        ]
        scores.sort(key=lambda pair: pair[1], reverse=True)
        return scores[:top_k]


if __name__ == "__main__":
    from corpus import DOCUMENTS, DOC_LABELS, QUERIES

    retriever = Bm25Retriever(DOCUMENTS)

    print(f"avg doc length: {retriever.avg_doc_length:.2f} tokens")
    print(f"doc lengths: {retriever.doc_lengths}\n")

    for query in QUERIES:
        print(f'Query: "{query}"')
        results = retriever.search(query)
        for doc_idx, score in results:
            print(f"  {score:.4f}  {DOC_LABELS[doc_idx]}")
        print()
