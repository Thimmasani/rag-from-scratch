---
title: 'Building RAG From Scratch, Part 3: Hybrid Retrieval With Reciprocal Rank Fusion'
description: 'Combining BM25 and embeddings on the same toy corpus — why you cannot just average their scores, how Reciprocal Rank Fusion works, and the real limitation it has: ties whenever two retrievers swap which document ranks first.'
pubDate: 'Oct 06 2026'
---

*Series: Building RAG From Scratch. Part 3 of ~6. Part 1 built BM25, Part 2 built embeddings, and both have now failed in specific, demonstrable ways on the same corpus. This part combines them.*

---

Part 1 and Part 2 each tell half the story. BM25 nails exact codes and fails on paraphrase ("car" vs "automobile"). Embeddings fix paraphrase and can route a bare ID to the wrong document when two systems share digits. Neither is strictly better — they fail on different inputs. The obvious next move is to run both and combine the results.

The obvious *wrong* way to combine them is averaging the raw scores.

---

## Why you can't just average the scores

Look at the actual numbers from Part 1 and Part 2 side by side, same corpus, same query:

| doc | BM25 | Embeddings |
|---|---|---|
| doc0 (query: "car oil change") | 2.4034 | 0.5903 |
| doc4 (same query) | 2.4034 | 0.6441 |

BM25 scores here range roughly 0 to 4.5 depending on term rarity and query length, unbounded above. Embedding cosine similarity is mathematically bounded in [-1, 1]. If you average these directly, BM25 dominates every single query just because its numbers are bigger — not because it's more relevant, just because of unit mismatch. You could normalize both to [0, 1] first, but then you're picking an arbitrary normalization scheme (min-max over what set of documents? per-query or per-corpus?) that has no principled justification and silently breaks the moment either retriever's score distribution shifts — a longer query, a different embedding model, a bigger corpus, and your hand-tuned weights are wrong again.

---

## Reciprocal Rank Fusion: fuse ranks, not scores

RRF sidesteps the scale problem by throwing away the raw scores entirely and working with *rank position* instead. Each retriever independently produces a ranked list; a document's fused score is the sum of `1 / (k + rank)` across every list it appears in:

```
RRF(d) = Σ over each retriever r:  1 / (k + rank_r(d))
```

`rank_r(d)` is the document's 1-indexed position in retriever `r`'s ranked list (1 = top result). `k` is a smoothing constant — commonly 60 — that controls how steeply the fusion favors top-ranked results over lower ones. A document ranked #1 by a retriever contributes `1/(k+1)`; ranked #10, it contributes `1/(k+10)`, a much smaller amount. A document that doesn't appear in a retriever's result list at all simply contributes nothing from that retriever — no penalty beyond "doesn't help."

```python
from collections import defaultdict

def reciprocal_rank_fusion(ranked_lists: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    fused_scores: dict[int, float] = defaultdict(float)
    for ranked_list in ranked_lists:
        for rank, doc_idx in enumerate(ranked_list):  # 0-indexed
            fused_scores[doc_idx] += 1 / (k + rank + 1)  # +1 -> rank 1 is top

    return sorted(fused_scores.items(), key=lambda pair: pair[1], reverse=True)


class HybridRetriever:
    def __init__(self, retrievers: list, k: int = 60):
        self.retrievers = retrievers
        self.k = k

    def search(self, query: str, top_k: int = 5, candidate_k: int = 10) -> list[tuple[int, float]]:
        ranked_lists = [
            [doc_idx for doc_idx, _score in retriever.search(query, top_k=candidate_k)]
            for retriever in self.retrievers
        ]
        return reciprocal_rank_fusion(ranked_lists, k=self.k)[:top_k]
```

No score normalization, no weighting scheme to tune. Both retrievers vote by rank, and RRF counts the votes.

---

## Running it on the original queries

Here's BM25, embeddings, and the fused RRF score together on the five queries from Part 1 and Part 2:

**Query: `"car oil change"`**

| doc | BM25 | Embed | RRF |
|---|---|---|---|
| doc0: car/oil/brakes, natural | 2.4034 | 0.5903 | 0.032522 |
| doc4: car/oil, overlaps doc0 | 2.4034 | **0.6441** | **0.032522** |
| doc5: "oil" × 11, spam | 1.3915 | 0.3865 | 0.031498 |

**Query: `"automobile brake service"`**

| doc | BM25 | Embed | RRF |
|---|---|---|---|
| doc1: automobile/oil/brakes | **4.1587** | **0.5647** | **0.032787** |
| doc0: car/oil/brakes, natural | 1.1432 | 0.4842 | 0.032258 |

Both retrievers already agreed doc1 was the answer here, so fusion just confirms it with the highest RRF score in the set (0.032787 — the maximum possible when a document is rank 1 in every retriever: `2 × 1/(60+1) = 0.032787`).

**Query: `"SKU-48213-B"`**

| doc | BM25 | Embed | RRF |
|---|---|---|---|
| doc2: exact SKU code | **1.7104** | **0.5710** | **0.032787** |

Same story — both retrievers independently nail this, fusion just reflects it.

---

## Where it gets interesting: ties

**Query: `"weekend weather"`**

| doc | BM25 | Embed | RRF |
|---|---|---|---|
| doc4: car dealership, says "weekend" | **1.7104** (rank 1) | 0.3379 (rank 2) | 0.032522 |
| doc3: actually about weather | 1.5868 (rank 2) | **0.3948** (rank 1) | 0.032522 |

Part 1 flagged this as lexical retrieval's worst failure (doc4 beats doc3 on BM25, for the wrong reason). Part 2 showed embeddings fix it decisively (doc3 beats doc4, 0.3948 to 0.3379 — not close). You'd expect the hybrid to clearly side with embeddings here. It doesn't: **both documents get the exact same RRF score**, 0.032522.

This isn't rounding. Work out the math: doc4 is rank 1 in BM25 and rank 2 in embeddings, so its RRF score is `1/(60+1) + 1/(60+2) = 0.032522`. doc3 is rank 2 in BM25 and rank 1 in embeddings — `1/(60+2) + 1/(60+1)`, the identical sum, just the two terms swapped. **Whenever two documents swap which one ranks #1 vs #2 between two equally-weighted retrievers, RRF always ties them exactly**, no matter how decisive either retriever's actual preference was. BM25's lead for doc4 was a coin flip (1.7104 vs 1.5868); embeddings' lead for doc3 was decisive (0.3948 vs 0.3379, about 17% ahead). RRF can't see that difference — it only sees "rank 1" and "rank 2," and a rank is a rank regardless of the score gap behind it.

---

## A sharper version of the same limitation

To see this isn't a one-off, I constructed a case specifically to probe it: a query mixing a semantic signal with an exact code string that belongs to the wrong document.

**Query: `"automobile SKU-48213-B service"`**

| doc | BM25 | Embed | RRF |
|---|---|---|---|
| doc1: automobile/oil/brakes (correct answer) | **2.9692** (rank 1) | 0.3644 (rank 2) | 0.032522 |
| doc2: exact SKU code (wrong answer here) | 1.7104 (rank 2) | **0.4920** (rank 1) | 0.032522 |

Embeddings alone get this wrong: the exact code string pulls doc2 to the top (0.4920 vs 0.3644), even though doc1 — matching "automobile" and "service" — is clearly the intended document. BM25 alone gets it right, and decisively (2.9692 vs 1.7104, doc1 almost double doc2's score). You'd hope fusion inherits BM25's confidence and corrects embeddings' mistake outright.

Instead: another exact tie. Same mechanism as "weekend weather" — doc1 is rank 1 in BM25 / rank 2 in embeddings, doc2 is the mirror image, so the sums are identical. The hybrid does make real progress here (doc2's wrongful outright lead collapses from a clear win to a tie), but it doesn't produce a clean correct answer. A tie-breaking rule (e.g. prefer the retriever with the larger score gap, or weight BM25 slightly higher when it has an exact token match) would resolve this in favor of doc1 — but that's an extra design decision on top of RRF, not something RRF gives you by default.

---

## What this means in practice

- **RRF is a strong default** precisely because it needs no score normalization or hand-tuned weights — just rank the same corpus with each retriever and sum reciprocal ranks. That's why it's a common first choice for combining lexical and semantic search in production systems.
- **It discards magnitude on purpose**, which is also its blind spot. Two retrievers that are "barely in disagreement" (BM25's near-tie on "weekend weather") and two that are "sharply in disagreement" (BM25's near-2x margin on the SKU query) get fused identically if the rank positions are the same. If you need the fusion to reflect *how confident* each retriever was, not just *which document it preferred*, RRF alone isn't enough — you'd want a learned fusion (weight each retriever's contribution based on historical reliability) or a reranking stage that looks at the actual query-document pairs again, with more signal than a rank number.
- **This is exactly what Part 4 (reranking) adds.** Hybrid retrieval here is a recall stage — cast a wide net combining two different notions of relevance. A cross-encoder reranker in the next part looks at each retrieved candidate against the query directly (not just independently-computed scores) and can break exactly the kind of tie RRF can't: "automobile SKU-48213-B service" against doc1 vs doc2, scored jointly, not fused from two separate opinions.

---

## What's next

Part 4 takes the fused candidates from this part's hybrid retriever and reranks them with a cross-encoder — a model that scores a query and a document together instead of independently, at the cost of being too slow to run over an entire corpus (which is why it reranks a short candidate list instead of replacing retrieval entirely).

Full code for this post, and the rest of the series: [github.com/Thimmasani/rag-from-scratch](https://github.com/Thimmasani/rag-from-scratch) — see `retrieval/rrf.py`, `compare_hybrid.py` (produces the tables above).
