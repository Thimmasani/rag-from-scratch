---
title: 'Building RAG From Scratch, Part 4: Cross-Encoder Reranking (and Where It Fails Too)'
description: 'Adding a cross-encoder reranking stage on top of Part 3''s hybrid retrieval — it cleanly fixes one RRF tie, but gets fooled by the exact same adversarial tricks that fooled embeddings, and by keyword stuffing that BM25 and embeddings both resisted.'
pubDate: 'Oct 08 2026'
---

*Series: Building RAG From Scratch. Part 4 of ~6. Part 3 left us with a specific, named gap: Reciprocal Rank Fusion ties two documents exactly whenever two retrievers swap which one ranks #1 vs #2, no matter how different their actual confidence was. Reranking is the standard fix. This part builds it, and is honest about where it still breaks.*

---

Part 3 ended with RRF unable to break two ties: "weekend weather" (BM25 barely preferred the wrong document, embeddings clearly preferred the right one, RRF averaged them into a dead heat) and a constructed case where an exact code string pulled embeddings toward the wrong document. The fix both cases call for is a model that looks at the query and a candidate document *together*, not two independent opinions fused after the fact. That's a cross-encoder.

This part builds one, reranks Part 3's hybrid candidates with it, and finds it fixes exactly one of those two problems cleanly — and introduces two new failures of its own that are worth taking seriously, not glossing over.

---

## Bi-encoder vs cross-encoder

Part 2's `EmbeddingRetriever` is a **bi-encoder**: the query and every document get embedded *independently* into fixed vectors, and relevance is a dot product between two vectors computed completely separately. That independence is the whole point — every document's embedding gets computed once, at index time, and reused for every future query. A new query costs one encode plus a matrix multiply against millions of precomputed vectors.

A **cross-encoder** feeds the query and one document into the model *together*, as a single combined input, and outputs one relevance score directly — no separate vectors, no dot product. The model's attention layers can directly compare query tokens against document tokens in context, which is strictly more expressive than comparing two independently-computed summaries. The cost is exactly what you'd expect: there's nothing to precompute. Every query-document pair needs its own full forward pass through the model.

```python
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, documents, candidate_indices, top_k=None):
        pairs = [(query, documents[idx]) for idx in candidate_indices]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(candidate_indices, scores), key=lambda pair: pair[1], reverse=True)
        return ranked[:top_k] if top_k else ranked
```

Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`, trained on the MS MARCO passage ranking dataset — real search queries paired with relevant and irrelevant passages. A standard, widely used reranking model, not something exotic.

### Why this can't replace retrieval

I measured this directly rather than asserting it. Same query, same CPU, comparing a bi-encoder search (one query encode + a dot product against *precomputed* document embeddings) against a cross-encoder rerank (a fresh forward pass per candidate, every time, since nothing can be cached):

| candidates (N) | bi-encoder search | cross-encoder rerank | ratio |
|---|---|---|---|
| 10 | 6.06 ms | 6.40 ms | 1.1x |
| 50 | 4.64 ms | 19.17 ms | 4.1x |
| 200 | 4.28 ms | 69.24 ms | 16.2x |

Bi-encoder search time stays roughly flat regardless of candidate count, because the expensive part (embedding every document) already happened once at index time — searching is just a matrix multiply against vectors that already exist. Cross-encoder time grows linearly with candidate count, because every single comparison requires a full model forward pass with no reuse. Run a cross-encoder over a corpus of 100,000 documents per query and you've made your retrieval system unusably slow. This is exactly why reranking exists as a *second stage*: let something fast (BM25, embeddings, or Part 3's hybrid fusion) narrow millions of documents down to a short candidate list — 10, 50, maybe 100 — then let the slow, expressive cross-encoder make the final call only on that short list.

---

## Case 1: it cleanly fixes the RRF tie

**Query: `"weekend weather"`** — Part 3's hybrid candidates, in RRF order, were `[doc4, doc3, doc0, doc5]`, an exact tie between doc4 (wrong) and doc3 (correct, actually about weather):

| doc | Reranked score |
|---|---|
| doc3: actually about weather | **-0.1569** |
| doc4: car dealership, says "weekend" | -7.3441 |

This is the case Part 3 was built to call out, and reranking resolves it decisively — not a near-tie, a 7-point gap on the model's raw logit scale. The cross-encoder can see "weekend weather" against the *full text* of each document simultaneously, rather than reconciling two independently-computed opinions after the fact. This is reranking doing exactly its intended job.

---

## Case 2: fooled by the same trap as embeddings

**Query: `"automobile SKU-48213-B service"`** — Part 3's hybrid candidates were `[doc1, doc2, doc0, doc5]`, another exact tie, this time between doc1 (correct) and doc2 (the SKU document, wrong for this query):

| doc | Reranked score |
|---|---|
| doc2: exact SKU code | **5.4989** |
| doc1: automobile/oil/brakes (correct answer) | -5.8122 |

The reranker doesn't just fail to fix this tie — it resolves it confidently in the *wrong* direction, picking doc2 by an 11-point margin. Part 2 showed embeddings alone made the same mistake (favoring doc2 at 0.4920 vs doc1's 0.3644), but only by a narrow margin, close enough that Part 3's RRF fusion pulled it to a tie. The cross-encoder, given the same literal exact-code string in both the query and doc2's text, latches onto that overlap even more decisively than the bi-encoder did. Whatever the MS MARCO training data taught this model about relevance, exact substring overlap is clearly a strong signal in it — strong enough to override the actual intent of the query.

---

## Case 3: fooled by keyword stuffing — a new failure, not inherited from Part 3

**Query: `"oil change"`** — this one doesn't even come from the RRF tie list. It's the keyword-stuffing query from Part 1, where BM25 (via term-frequency saturation) and embeddings (Part 2) both correctly kept the spam document, doc5, from winning. Hybrid's RRF order was `[doc0, doc5, doc4, doc1]` — doc0, the natural phrasing, on top.

| doc | Reranked score |
|---|---|
| doc5: "oil" × 11, keyword-stuffed spam | **5.1681** |
| doc4: car/oil, overlaps doc0 | 4.5360 |
| doc0: natural phrasing (correct answer) | 1.5375 |

The reranker puts the spam document **first**, ahead of both doc4 and the correct doc0. I checked this in isolation (just doc0 vs doc5, no other candidates in the mix) and got the same result: 1.5375 vs 5.1681. This isn't a candidate-list artifact — the model genuinely scores eleven repetitions of "oil" as more relevant to "oil change" than one natural, correctly-phrased sentence. Two completely different defenses against keyword stuffing — BM25's explicit saturation term and the bi-encoder's meaning-based embedding — both held up in earlier parts. This cross-encoder has no equivalent defense; term density in the passage appears to matter a lot to it.

---

## Case 4: a partial, not total, improvement

Worth being precise about one case that isn't a clean win or a clean loss. On the extended corpus from Part 2 (the invoice/SKU domain-collision setup), querying the bare ID `"INV-48213-B"`:

| doc | Hybrid (RRF) rank | Reranked rank | Reranked score |
|---|---|---|---|
| doc2: exact SKU code (wrong answer) | 1st | 1st | **5.5062** |
| doc6: invoice number (correct answer) | 4th (last candidate) | 2nd | 0.2341 |

Reranking doesn't fix this one either — doc2 still wins. But doc6 moves from dead last among the four candidates up to second place. That's a real, if incomplete, improvement: the reranker's judgment of doc6's relevance is better than the hybrid fusion's was, even though it's not good enough to overturn doc2's lead here. Not every case resolves to "fixed" or "still broken" — some just get somewhat better.

---

## What this means in practice

- **Reranking earns its cost on genuine ties between reasonably-matched signals** — exactly the "weekend weather" pattern, where two retrievers each have a real but limited signal and disagree about priority. Joint attention over the actual query and document text breaks that kind of tie better than fusing independently-computed ranks ever could.
- **A reranker is still a trained model with its own biases, not a ground-truth oracle.** This one appears to weight literal term overlap heavily — enough to be fooled by the same exact-code injection that fooled embeddings (and more confidently), and by keyword density that neither BM25 nor embeddings fell for. "Add a reranker" is not a universal correctness upgrade; it changes *which* biases your pipeline has, and in this case, for these specific adversarial queries, in a worse direction.
- **The adversarial cases in this series were constructed to be hard on purpose.** Real queries are rarely as deliberately confusing as "automobile SKU-48213-B service" or eleven repetitions of "oil." On the ordinary queries from Parts 1-3, this same reranker performs exactly as expected. The lesson isn't "don't use cross-encoders" — it's that evaluating a retrieval pipeline needs adversarial, not just average-case, test queries, because average-case metrics will hide exactly the failure modes this part surfaced.
- **Fixing cases 2 and 3 for real needs something other than "a smarter model."** Exact-identifier confusion is better solved by format-aware query understanding (detect ID-shaped tokens, route them through exact-match lookup regardless of what any learned model thinks) or metadata filtering, not by hoping a bigger reranker eventually learns not to be fooled by a string it's seen verbatim in the query.

---

## What's next

Part 5 moves to the generation layer: taking retrieved (and now reranked) context and actually constructing a prompt for an LLM — grounding, citation formatting, and what the system does when retrieval legitimately finds nothing relevant. The queries this part flagged as broken (exact-code injection, keyword stuffing) turn out to matter even more downstream, once there's a generated answer a user actually reads.

Full code for this post, and the rest of the series: [github.com/Thimmasani/rag-from-scratch](https://github.com/Thimmasani/rag-from-scratch) — see `retrieval/reranker.py`, `compare_reranked.py` (produces the tables above).
