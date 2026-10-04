"""
TF-IDF from scratch, no sklearn.

Implements:
  - tokenization (simple, lowercase + split; good enough for the toy corpus)
  - term frequency (TF)
  - inverse document frequency (IDF), smoothed variant used by scikit-learn
    so numbers are easy to sanity-check against a familiar library later
  - TF-IDF vector construction
  - cosine similarity ranking

Formulas (smoothed IDF, matches scikit-learn's default):
    tf(t, d)  = count of term t in document d
    idf(t)    = ln((1 + N) / (1 + df(t))) + 1
        N     = total number of documents
        df(t) = number of documents containing term t
    tfidf(t, d) = tf(t, d) * idf(t)

Cosine similarity between query vector q and document vector d:
    cos(q, d) = (q . d) / (||q|| * ||d||)
"""

import math
import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    """Lowercase + extract word tokens. Keeps alphanumerics and hyphens so
    tokens like 'SKU-48213-B' survive as a single token (relevant for the
    exact-match example in the corpus)."""
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


class TfidfRetriever:
    def __init__(self, documents: list[str]):
        self.documents = documents
        self.tokenized_docs = [tokenize(doc) for doc in documents]
        self.n_docs = len(documents)

        # vocabulary = sorted unique terms across the whole corpus
        vocab_set = set()
        for tokens in self.tokenized_docs:
            vocab_set.update(tokens)
        self.vocab = sorted(vocab_set)
        self.term_to_idx = {term: i for i, term in enumerate(self.vocab)}

        self.idf = self._compute_idf()
        self.doc_vectors = [self._vectorize(tokens) for tokens in self.tokenized_docs]

    def _compute_idf(self) -> dict[str, float]:
        df = Counter()
        for tokens in self.tokenized_docs:
            for term in set(tokens):
                df[term] += 1

        idf = {}
        for term in self.vocab:
            idf[term] = math.log((1 + self.n_docs) / (1 + df[term])) + 1
        return idf

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        """Return a sparse {term: tfidf_weight} vector for a token list."""
        tf = Counter(tokens)
        vec = {}
        for term, count in tf.items():
            if term in self.idf:  # ignore out-of-vocabulary query terms
                vec[term] = count * self.idf[term]
        return vec

    @staticmethod
    def _cosine_sim(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
        shared_terms = vec_a.keys() & vec_b.keys()
        dot = sum(vec_a[t] * vec_b[t] for t in shared_terms)

        norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
        norm_b = math.sqrt(sum(v * v for v in vec_b.values()))

        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """Return [(doc_index, score), ...] sorted by descending score."""
        query_vec = self._vectorize(tokenize(query))
        scores = [
            (i, self._cosine_sim(query_vec, doc_vec))
            for i, doc_vec in enumerate(self.doc_vectors)
        ]
        scores.sort(key=lambda pair: pair[1], reverse=True)
        return scores[:top_k]


if __name__ == "__main__":
    from corpus import DOCUMENTS, DOC_LABELS, QUERIES

    retriever = TfidfRetriever(DOCUMENTS)

    print(f"Vocabulary ({len(retriever.vocab)} terms): {retriever.vocab}\n")

    for query in QUERIES:
        print(f'Query: "{query}"')
        results = retriever.search(query)
        for doc_idx, score in results:
            print(f"  {score:.4f}  {DOC_LABELS[doc_idx]}")
        print()
