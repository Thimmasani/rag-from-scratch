"""
The full RAG pipeline, start to finish: Parts 1-5 wired together.

    query
      -> BM25 (Part 1) + embeddings (Part 2), run independently
      -> Reciprocal Rank Fusion (Part 3)
      -> cross-encoder reranking on the fused candidates (Part 4)
      -> generation with abstention + citations (Part 5)

Every stage is a module built from scratch earlier in this series -- no
LangChain retriever/chain abstractions assembling this for you.
"""

from corpus import DOCUMENTS, DOC_LABELS, QUERIES
from bm25 import Bm25Retriever
from embeddings import EmbeddingRetriever
from rrf import HybridRetriever
from reranker import CrossEncoderReranker
from generation import Generator, format_with_citations


def run_pipeline(
    query: str,
    documents: list[str],
    doc_labels: list[str],
    hybrid: HybridRetriever,
    reranker: CrossEncoderReranker,
    generator: Generator,
    retrieve_k: int = 4,
    generate_k: int = 2,
):
    hybrid_candidates = [idx for idx, _score in hybrid.search(query, top_k=retrieve_k)]
    reranked = reranker.rerank(query, documents, hybrid_candidates, top_k=generate_k)
    top_doc_indices = [idx for idx, _score in reranked]
    context_chunks = [documents[idx] for idx in top_doc_indices]

    result = generator.answer(query, context_chunks)

    print(f'Query: "{query}"')
    print(f"  Hybrid candidates:  {[doc_labels[i].split(':')[0] for i in hybrid_candidates]}")
    print(f"  After reranking:    {[doc_labels[i].split(':')[0] for i in top_doc_indices]}")
    print(f"  Grounded:           {result['grounded']}")
    if result["grounded"]:
        # remap citation indices (into context_chunks) back to doc labels
        labeled = [
            (sentence, [doc_labels[top_doc_indices[i]] for i in sources])
            for sentence, sources in result["citations"]
        ]
        print("  Answer:")
        for sentence, sources in labeled:
            print(f"    {sentence}  <- {sources}")
    else:
        print(f"  Answer: {result['answer']}")
    print()


if __name__ == "__main__":
    bm25 = Bm25Retriever(DOCUMENTS)
    emb = EmbeddingRetriever(DOCUMENTS)
    hybrid = HybridRetriever([bm25, emb])
    reranker = CrossEncoderReranker()
    generator = Generator()

    for query in QUERIES:
        run_pipeline(query, DOCUMENTS, DOC_LABELS, hybrid, reranker, generator)

    # A query with no good answer anywhere in this corpus -- the pipeline
    # should abstain rather than force an answer out of irrelevant context.
    run_pipeline("What is the capital of France?", DOCUMENTS, DOC_LABELS, hybrid, reranker, generator)
