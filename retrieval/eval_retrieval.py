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

import math

from corpus import DOCUMENTS, GROUND_TRUTH
from bm25 import Bm25Retriever
from embeddings import EmbeddingRetriever
from rrf import HybridRetriever
from reranker import CrossEncoderReranker


def recall_at_k(ranked_doc_indices: list[int], correct_doc: int, k: int) -> bool:
    return correct_doc in ranked_doc_indices[:k]


def precision_at_k(ranked_doc_indices: list[int], correct_doc: int, k: int) -> float:
    """With exactly one relevant document in this ground truth, this is
    1/k if that document is in the top-k, else 0 -- i.e. an exact linear
    rescaling of recall_at_k (precision@k = recall@k / k) under THIS
    single-relevant-document setup. It does not add new ranking signal
    here; it's included because interview-style discussions name it
    separately, and the article states this equivalence explicitly rather
    than implying precision@k shows something recall@k doesn't."""
    return (1.0 / k) if recall_at_k(ranked_doc_indices, correct_doc, k) else 0.0


def mrr(ranked_doc_indices: list[int], correct_doc: int) -> float:
    """1/rank of the correct document (1-indexed), or 0 if it's absent
    from the ranked list entirely."""
    for rank, doc_idx in enumerate(ranked_doc_indices, start=1):
        if doc_idx == correct_doc:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_doc_indices: list[int], correct_doc: int, k: int) -> float:
    """Implemented generally (works for graded relevance too), but with
    this ground truth's BINARY relevance (one correct doc, everything
    else irrelevant), NDCG@k reduces to 1/log2(rank+1) if the correct doc
    is within the top-k, else 0 -- it moves with MRR here, not
    independently. The graded-relevance case this metric is designed for
    isn't exercised by this ground truth; stated honestly rather than
    implied."""
    for rank, doc_idx in enumerate(ranked_doc_indices[:k], start=1):
        if doc_idx == correct_doc:
            dcg = 1.0 / math.log2(rank + 1)
            ideal_dcg = 1.0 / math.log2(1 + 1)  # best case: correct doc at rank 1
            return dcg / ideal_dcg
    return 0.0


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


def evaluate_stage_extended(name: str, rank_fn, queries_and_answers: dict[str, int], k: int):
    """precision@k, MRR, and NDCG@k for one stage -- extends the recall@k
    table above with the metrics discussed in the interview article's Q7,
    computed for real against this series' own ground truth."""
    n = len(queries_and_answers)
    total_precision = 0.0
    total_mrr = 0.0
    total_ndcg = 0.0
    for query, correct_doc in queries_and_answers.items():
        ranked = rank_fn(query)
        total_precision += precision_at_k(ranked, correct_doc, k)
        total_mrr += mrr(ranked, correct_doc)
        total_ndcg += ndcg_at_k(ranked, correct_doc, k)

    print(f"--- {name} (k={k}) ---")
    print(f"  precision@{k}: {total_precision / n:.4f}")
    print(f"  MRR:           {total_mrr / n:.4f}")
    print(f"  NDCG@{k}:       {total_ndcg / n:.4f}")
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

    print("=== precision@k, MRR, NDCG@k (k=1) ===\n")
    evaluate_stage_extended(
        "BM25 alone (Part 1)",
        lambda q: [idx for idx, _ in bm25.search(q, top_k=max(k_values))],
        answerable,
        1,
    )
    evaluate_stage_extended(
        "Embeddings alone (Part 2)",
        lambda q: [idx for idx, _ in emb.search(q, top_k=max(k_values))],
        answerable,
        1,
    )
    evaluate_stage_extended(
        "Hybrid RRF (Part 3)",
        lambda q: [idx for idx, _ in hybrid.search(q, top_k=max(k_values))],
        answerable,
        1,
    )
    evaluate_stage_extended("Hybrid + reranked (Part 4)", reranked_order, answerable, 1)
