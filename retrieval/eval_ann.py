"""
Vector search at scale: evaluation driver, Part 8 of the RAG-from-scratch
series.

Two experiments:
  1. Recall vs. latency: brute force vs. HNSW at a few corpus sizes,
     measuring HNSW's recall@k against brute force's EXACT result as
     ground truth (not against any hand-labeled "correct" answer -- at
     this scale, "correct" means "what exhaustive search would have
     returned").
  2. Metadata filtering: pre-filter vs. post-filter under a deliberately
     selective filter (the ~2% "clearance" tag from ann_corpus.py),
     showing how often post-filtering returns fewer than top_k usable
     results.
"""

import time

import numpy as np
from sentence_transformers import SentenceTransformer

from ann_corpus import generate_synthetic_corpus
from ann_search import BruteForceIndex, HnswIndex, pre_filter_search, post_filter_search

QUERIES = [
    "durable wireless headphones with a warranty",
    "lightweight yoga mat for home workouts",
    "affordable cookbook with easy recipes",
    "compact desk lamp for a small room",
    "remote control car for kids",
]


def recall_latency_sweep(model: SentenceTransformer, sizes: list[int], top_k: int = 10):
    print("=== 1. Recall vs. latency: brute force vs. HNSW ===\n")
    query_embeddings = model.encode(QUERIES, normalize_embeddings=True)

    for n in sizes:
        docs, _meta = generate_synthetic_corpus(n, seed=42)
        doc_embeddings = model.encode(docs, normalize_embeddings=True, show_progress_bar=False)

        brute = BruteForceIndex(doc_embeddings)
        hnsw = HnswIndex(doc_embeddings, ef_search=50)

        brute_times, hnsw_times = [], []
        recall_hits = 0
        for q_emb in query_embeddings:
            t0 = time.perf_counter()
            brute_result = brute.search(q_emb, top_k)
            t1 = time.perf_counter()
            hnsw_result = hnsw.search(q_emb, top_k)
            t2 = time.perf_counter()

            brute_times.append((t1 - t0) * 1000)
            hnsw_times.append((t2 - t1) * 1000)

            brute_ids = {idx for idx, _ in brute_result}
            hnsw_ids = {idx for idx, _ in hnsw_result}
            recall_hits += len(brute_ids & hnsw_ids) / top_k

        avg_recall = recall_hits / len(QUERIES)
        print(
            f"  n={n:>6}  brute-force: {sum(brute_times)/len(brute_times):.3f}ms  "
            f"HNSW: {sum(hnsw_times)/len(hnsw_times):.3f}ms  "
            f"HNSW recall@{top_k} vs. exact: {avg_recall:.3f}"
        )
    print()


def ef_search_sweep(model: SentenceTransformer, n: int = 10000, top_k: int = 10):
    """ef_search is the knob that's supposed to trade recall for latency.
    Sweep it at the largest tested size to check whether that trade-off
    actually shows up on this fixture."""
    print(f"=== ef_search sweep at n={n} (does the recall/latency trade-off show up?) ===\n")
    docs, _meta = generate_synthetic_corpus(n, seed=42)
    doc_embeddings = model.encode(docs, normalize_embeddings=True, show_progress_bar=False)
    brute = BruteForceIndex(doc_embeddings)
    query_embeddings = model.encode(QUERIES, normalize_embeddings=True)

    for ef in [1, 5, 10, 50, 200]:
        hnsw = HnswIndex(doc_embeddings, ef_search=ef)
        times = []
        recall_hits = 0
        for q_emb in query_embeddings:
            t0 = time.perf_counter()
            hnsw_result = hnsw.search(q_emb, top_k)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
            brute_ids = {idx for idx, _ in brute.search(q_emb, top_k)}
            hnsw_ids = {idx for idx, _ in hnsw_result}
            recall_hits += len(brute_ids & hnsw_ids) / top_k
        print(f"  ef_search={ef:>4}  latency: {sum(times)/len(times):.4f}ms  recall@{top_k}: {recall_hits/len(QUERIES):.3f}")
    print()


def metadata_filtering_demo(model: SentenceTransformer, n: int = 5000, top_k: int = 10, candidate_k: int = 10):
    print(f"=== 2. Metadata filtering: pre-filter vs. post-filter (n={n}) ===\n")
    docs, meta = generate_synthetic_corpus(n, seed=42)
    doc_embeddings = model.encode(docs, normalize_embeddings=True, show_progress_bar=False)
    hnsw = HnswIndex(doc_embeddings, ef_search=50)

    n_clearance = sum(1 for m in meta if m["category"] == "clearance")
    print(f"{n_clearance}/{n} documents ({n_clearance/n:.1%}) tagged category='clearance'\n")

    def is_clearance(m: dict) -> bool:
        return m["category"] == "clearance"

    query_embeddings = model.encode(QUERIES, normalize_embeddings=True)

    pre_short = 0
    post_short = 0
    post_empty = 0
    for q_emb in query_embeddings:
        pre_results = pre_filter_search(doc_embeddings, meta, q_emb, is_clearance, top_k)
        post_results = post_filter_search(hnsw, q_emb, meta, is_clearance, top_k, candidate_k)

        if len(pre_results) < top_k:
            pre_short += 1
        if len(post_results) < top_k:
            post_short += 1
        if len(post_results) == 0:
            post_empty += 1

        print(
            f"  query: {len(pre_results)} pre-filter results, "
            f"{len(post_results)} post-filter results (requested top_k={top_k}, candidate_k={candidate_k})"
        )

    print(
        f"\n  Pre-filter: {pre_short}/{len(QUERIES)} queries returned fewer than top_k results "
        f"(only when fewer than top_k matching docs exist at all)."
    )
    print(
        f"  Post-filter: {post_short}/{len(QUERIES)} queries returned fewer than top_k results, "
        f"{post_empty}/{len(QUERIES)} returned ZERO -- because the filter is applied AFTER narrowing "
        f"to only {candidate_k} candidates, and most of those candidates aren't clearance items."
    )


if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")
    recall_latency_sweep(model, sizes=[1000, 5000, 10000])
    ef_search_sweep(model, n=10000)
    metadata_filtering_demo(model, n=5000)
