"""
Retrieval evaluation: recall@k, Part 6 of the RAG-from-scratch series.

recall@k for a single query = 1 if the ground-truth document appears
anywhere in the top-k results, else 0. Averaged across queries, this is
the standard first metric for a retrieval system: "how often is the right
answer even in the candidate list we'd hand to the next stage?" It says
nothing about whether the right answer is ranked #1 vs #k -- that's what
precision/MRR/NDCG are for -- but it's the right metric for "did we lose
the answer before generation even got a chance."

This evaluates every retrieval stage built in this series against the
same ground truth (corpus.py's GROUND_TRUTH), on the same queries, so the
stages are directly comparable:
  - BM25 alone (Part 1)
  - Embeddings alone (Part 2)
  - Hybrid RRF fusion (Part 3)
  - Hybrid + cross-encoder reranking (Part 4)

Queries with no correct document (GROUND_TRUTH value None, e.g. the
France query) are excluded from recall@k -- recall@k measures "did we
find the right document," which is undefined when there isn't one. Part
6's groundedness eval (eval_generation.py) is what covers the abstention
case instead.
"""

from corpus import DOCUMENTS, GROUND_TRUTH
from bm25 import Bm25Retriever
from embeddings import EmbeddingRetriever
from rrf import HybridRetriever
from reranker import CrossEncoderReranker


def recall_at_k(ranked_doc_indices: list[int], correct_doc: int, k: int) -> bool:
    return correct_doc in ranked_doc_indices[:k]


def evaluate_stage(name: str, rank_fn, queries_and_answers: dict[str, int], k_values: list[int]):
    print(f"--- {name} ---")
    for k in k_values:
        hits = 0
        for query, correct_doc in queries_and_answers.items():
            ranked = rank_fn(query)
            if recall_at_k(ranked, correct_doc, k):
                hits += 1
        recall = hits / len(queries_and_answers)
        print(f"  recall@{k}: {recall:.2f}  ({hits}/{len(queries_and_answers)})")
    print()


if __name__ == "__main__":
    bm25 = Bm25Retriever(DOCUMENTS)
    emb = EmbeddingRetriever(DOCUMENTS)
    hybrid = HybridRetriever([bm25, emb])
    reranker = CrossEncoderReranker()

    # Exclude the abstention-only query (no correct doc to find)
    answerable = {q: doc for q, doc in GROUND_TRUTH.items() if doc is not None}

    k_values = [1, 2, 3]

    evaluate_stage(
        "BM25 alone (Part 1)",
        lambda q: [idx for idx, _ in bm25.search(q, top_k=max(k_values))],
        answerable,
        k_values,
    )
    evaluate_stage(
        "Embeddings alone (Part 2)",
        lambda q: [idx for idx, _ in emb.search(q, top_k=max(k_values))],
        answerable,
        k_values,
    )
    evaluate_stage(
        "Hybrid RRF (Part 3)",
        lambda q: [idx for idx, _ in hybrid.search(q, top_k=max(k_values))],
        answerable,
        k_values,
    )

    def reranked_order(query: str) -> list[int]:
        candidates = [idx for idx, _ in hybrid.search(query, top_k=4)]
        reranked = reranker.rerank(query, DOCUMENTS, candidates)
        return [idx for idx, _ in reranked]

    evaluate_stage("Hybrid + reranked (Part 4)", reranked_order, answerable, k_values)
