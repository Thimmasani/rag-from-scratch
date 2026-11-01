---
title: 'Building RAG From Scratch, Part 8: Vector Search at Scale'
description: "Brute-force search over embeddings is basically free at 4 documents. This part measures what happens at 1,000-10,000 documents with HNSW (an approximate nearest-neighbor index), and what metadata filtering actually costs when combined with it."
pubDate: 'Oct 22 2026'
---

*Series: Building RAG From Scratch. Part 8. Part 2's `EmbeddingRetriever.search()` is `doc_embeddings @ query_embedding`, a full sort over every document — exact and, on a 6-document corpus, effectively instant. This part asks what happens once that stops being true, and what it costs to also filter by metadata at the same time.*

---

Every retriever built so far in this series — `TfidfRetriever`, `Bm25Retriever`, `EmbeddingRetriever` — does an exhaustive scan: score every document, sort, take the top-k. On 6 documents that's free. At real scale it isn't, and that's the problem approximate nearest neighbor (ANN) search exists to solve.

**One honest note before the dependency discussion below:** this is the first part in the series that doesn't implement its core algorithm from scratch. BM25 and RRF are closed-form formulas, hand-rollable in under 100 lines. HNSW is a stateful, tunable, multi-layer graph-construction algorithm — correctly implementing it is its own multi-week project, not a blog-post-sized experiment, and a buggy hand-rolled version would misrepresent the real thing. This part uses [`hnswlib`](https://github.com/nmslib/hnswlib), a minimal, single-algorithm, pip-installable, local, no-API-key library — one focused dependency for the one specific structure being measured, not a general-purpose vector-DB client pulling in far more surface area than this topic needs.

---

## A synthetic corpus big enough to matter

`retrieval/ann_corpus.py` deterministically generates `n` short product-description documents (template-based: category, adjective, feature, tagged with `category` and `date` metadata), seeded for reproducibility without committing a large data file. One document in 50 is tagged `category="clearance"` — a deliberately selective ~2% filter, used in the metadata-filtering experiment below.

```python
def generate_synthetic_corpus(n: int, seed: int = 42) -> tuple[list[str], list[dict]]:
    ...
    # ~2% tagged category="clearance", the rest spread across
    # electronics/home/sports/books/toys with a random month.
```

Two index types, both built from the same precomputed embeddings (`sentence-transformers/all-MiniLM-L6-v2`, Part 2's model):

```python
class BruteForceIndex:
    """Exact: doc_embeddings @ query_embedding, full sort. The baseline."""
    def search(self, query_embedding, top_k):
        scores = self.doc_embeddings @ query_embedding
        ranked = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in ranked]

class HnswIndex:
    """Approximate: a hierarchical graph, descended coarse-to-fine."""
    def __init__(self, doc_embeddings, ef_search=50):
        self.index = hnswlib.Index(space="cosine", dim=384)
        self.index.init_index(max_elements=len(doc_embeddings), ef_construction=200, M=16)
        self.index.add_items(doc_embeddings, np.arange(len(doc_embeddings)))
        self.index.set_ef(ef_search)

    def search(self, query_embedding, top_k):
        labels, distances = self.index.knn_query(query_embedding, k=top_k)
        return [(int(l), float(1.0 - d)) for l, d in zip(labels[0], distances[0])]
```

---

## 1. Brute force scales with corpus size. HNSW doesn't.

Five test queries, averaged, at three corpus sizes:

| corpus size | brute-force latency | HNSW latency (ef_search=50) | HNSW recall@10 vs. exact |
|---|---|---|---|
| 1,000 | 0.063ms | 0.070ms | 1.000 |
| 5,000 | 0.238ms | 0.085ms | 1.000 |
| 10,000 | 0.496ms | 0.104ms | 1.000 |

Brute-force latency grows roughly linearly with corpus size, exactly as its O(N) description predicts — it's already ~8x slower at 10,000 documents than at 1,000. HNSW's latency barely moves (0.070ms → 0.104ms, a 1.5x increase over a 10x larger corpus), because descending a graph coarse-to-fine doesn't require touching every vector. At corpus sizes like these, the gap is still small in absolute terms — both are sub-millisecond — but the *trend* is the entire point: keep scaling the corpus up by another couple of orders of magnitude (which real production vector search routinely does) and brute force keeps scaling linearly while HNSW keeps staying close to flat.

## 2. The recall/latency trade-off didn't show up — and that's worth reporting honestly

ANN's textbook pitch is "trade a little recall for a lot of speed," controlled by `ef_search` (how many candidates the graph search explores per query). Sweeping it directly, at the largest corpus size:

| ef_search | latency | recall@10 vs. exact |
|---|---|---|
| 1 | 0.044ms | 1.000 |
| 5 | 0.053ms | 1.000 |
| 10 | 0.050ms | 1.000 |
| 50 | 0.113ms | 1.000 |
| 200 | 0.273ms | 1.000 |

Recall stays perfect even at `ef_search=1`, the most aggressive (fastest, least thorough) setting tested. This isn't a bug — it's a property of this specific fixture: the synthetic corpus's five categories use distinct enough vocabulary that even a minimal graph search finds the true nearest neighbors every time. The recall/latency trade-off is real (it's well documented at larger scale and on harder, less cleanly separable embeddings — tightening `ef_search` eventually does cost recall), but it genuinely did not materialize on this fixture at this scale. Reporting a clean trade-off here anyway, because that's the textbook story, would be exactly the kind of unverified claim this series has tried not to make. The honest finding is narrower: HNSW's latency advantage held up regardless of `ef_search`, and this fixture wasn't hard enough to expose the recall cost that forces `ef_search` to matter in practice.

## 3. Pre-filter vs. post-filter: a selective metadata filter breaks one of them

Using the ~2% `category="clearance"` filter against a 5,000-document corpus, five queries, `top_k=10`:

```python
def pre_filter_search(doc_embeddings, metadata, query_embedding, filter_fn, top_k):
    matching = [i for i, m in enumerate(metadata) if filter_fn(m)]
    sub_embeddings = doc_embeddings[matching]
    scores = sub_embeddings @ query_embedding
    ...  # search ONLY within the matching subset

def post_filter_search(index, query_embedding, metadata, filter_fn, top_k, candidate_k):
    candidates = index.search(query_embedding, candidate_k)
    return [(i, s) for i, s in candidates if filter_fn(metadata[i])][:top_k]
```

| | pre-filter | post-filter (candidate_k=10) |
|---|---|---|
| queries returning fewer than 10 results | 0/5 | **5/5** |
| queries returning **zero** results | 0/5 | **4/5** |

Pre-filtering narrows to the matching subset *before* searching, so it always returns up to `top_k` results as long as that many matching documents exist anywhere in the corpus — which they do here (100 clearance items, far more than 10). Post-filtering searches the top 10 candidates first, *then* drops whichever fail the filter — and since only ~2% of the corpus is tagged clearance, the odds that any of the top 10 nearest-neighbor candidates happen to be clearance items are low. Four of five test queries got back zero usable results under post-filtering, not because no matching document existed, but because none of them made it into the narrow candidate window the filter was applied to.

Raising `candidate_k` (searching a wider candidate pool before filtering) would recover some of this at the cost of more work per query — the real, concrete version of the trade-off the interview article's Q17 describes in general terms: pre-filtering needs an index that supports efficient filtered search, post-filtering is simpler but risks starving the result set exactly when the filter is most selective.

---

## What this means in practice

- **Brute force genuinely is fine at small-to-moderate scale.** Every retriever in Parts 1-6 using exhaustive scan was never a shortcut being taken, it was the correct choice at that scale. The crossover point where an ANN index starts to matter is a real measurement to make on your own data, not an assumption.
- **A recall/latency trade-off being real in general doesn't mean it will show up on your specific corpus.** This fixture's categories were cleanly separable enough that even an aggressive `ef_search=1` never missed the true nearest neighbors. Harder, more semantically overlapping corpora are exactly where that slack disappears — worth testing directly rather than assuming the textbook trade-off curve applies uniformly.
- **A selective metadata filter can silently starve post-filtering.** 4 of 5 queries returning zero results isn't a hypothetical edge case, it's what actually happened on the very first selective filter tested. If your production filter is ever this selective (a permissions filter restricting to one tenant's documents, for instance), pre-filtering or a much larger `candidate_k` isn't an optimization, it's the difference between a working feature and a silently empty response.

---

## What's next

[Part 9](/blog/rag-from-scratch-09-multihop-retrieval/) goes back to the original 6-document scale of thinking, but with a different retrieval pattern entirely: questions that need facts from *two* separate documents at once, which a single retrieval call over the whole question can't reliably assemble.

Full code for this post: [github.com/Thimmasani/rag-from-scratch](https://github.com/Thimmasani/rag-from-scratch) — see `retrieval/ann_corpus.py`, `ann_search.py`, `eval_ann.py`.
