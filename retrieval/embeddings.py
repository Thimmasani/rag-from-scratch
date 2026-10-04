"""
Semantic retrieval via sentence embeddings, Part 2 of the RAG-from-scratch
series.

Unlike tfidf.py/bm25.py, this isn't "from scratch" in the sense of
hand-deriving the model -- training a competitive embedding model yourself
is a different project entirely. What *is* from scratch here is the
retrieval logic itself: we embed documents and queries, then rank by cosine
similarity using plain vector math, no vector-DB abstraction, no
LangChain Retriever hiding what's actually happening.

Model: sentence-transformers/all-MiniLM-L6-v2 (384-dim), a small, fast,
widely used sentence embedding model, good enough to demonstrate the
behavior differences from lexical retrieval without needing a GPU.

Cosine similarity when both vectors are L2-normalized reduces to a plain
dot product, which is what we use below -- this is the same normalization
trick that makes MIPS-based ANN indexes (FAISS, HNSW) usable for
cosine-style semantic search (see writing/02_embeddings.md for the full
explanation).
"""

from sentence_transformers import SentenceTransformer


class EmbeddingRetriever:
    def __init__(self, documents: list[str], model_name: str = "all-MiniLM-L6-v2"):
        self.documents = documents
        self.model = SentenceTransformer(model_name)
        # normalize_embeddings=True -> vectors are unit length, so dot
        # product == cosine similarity.
        self.doc_embeddings = self.model.encode(documents, normalize_embeddings=True)

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        query_embedding = self.model.encode(query, normalize_embeddings=True)
        scores = self.doc_embeddings @ query_embedding
        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
        return [(idx, float(score)) for idx, score in ranked[:top_k]]


if __name__ == "__main__":
    from corpus import DOCUMENTS, DOC_LABELS, QUERIES

    retriever = EmbeddingRetriever(DOCUMENTS)

    for query in QUERIES:
        print(f'Query: "{query}"')
        for doc_idx, score in retriever.search(query):
            print(f"  {score:.4f}  {DOC_LABELS[doc_idx]}")
        print()
