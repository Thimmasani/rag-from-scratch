---
title: 'Building RAG From Scratch, Part 6: Evaluating the Whole Pipeline'
description: "recall@k across every retrieval stage built in this series, and an LLM-as-judge groundedness check tested directly against Part 5's real fabrication — with a genuinely surprising result: adding reranking made recall@1 worse, not better."
pubDate: 'Oct 12 2026'
---

*Series: Building RAG From Scratch. Part 6 of 6 originally planned parts — the series continued with Parts 7-10, linked below. Parts 1-5 built lexical retrieval, embeddings, hybrid fusion, reranking, and generation. This part measures all of it against ground truth, instead of trusting that each stage's design intent translated into an actual improvement.*

---

Every part in this series has reported specific findings by hand: this query failed here, that fix worked there. That's useful, but it's anecdotal — individual examples, not a measurement across the whole system. This part builds the two metrics a real RAG pipeline needs: **recall@k** for retrieval (did we find the right document at all?) and a **groundedness check** for generation (is the generated claim actually true according to its source?). Both are tested against ground truth established earlier in the series, not invented for this part.

---

## recall@k across every retrieval stage

recall@k for one query is 1 if the correct document appears anywhere in the top-k results, 0 otherwise. Averaged across queries, it answers a narrower but critical question: before worrying about rank order, did the retrieval stage even keep the right document in play? Every query used across Parts 1-4 has an established correct answer (discussed explicitly in each part's article), so this isn't a new labeled set built for this part — it's the ground truth the series has already been using, made explicit and measured systematically.

**A note on sample size, stated up front rather than left for a reader to catch:** the first version of this evaluation ran on just 6 queries. With n=6, recall@1 moves in increments of 1/6 — about 17 percentage points — *per query*. A two-query swing reads as a dramatic 33-point regression, which risks overstating how precisely a 6-item sample can measure anything. So the ground truth below adds 12 more queries against the exact same 6-document corpus — no new documents, no change to the original 6 queries — chosen by plain vocabulary correspondence to one specific document, not chosen to push the result either way:

```python
GROUND_TRUTH = {
    # the original 6, unchanged
    "car oil change": 0,
    "automobile brake service": 1,
    "SKU-48213-B": 2,
    "weekend weather": 3,
    "oil change": 0,
    "automobile SKU-48213-B service": 1,
    # 12 more, same corpus, ordinary non-adversarial queries
    "brake pads replacement": 0,
    "engine oil service": 1,
    "SKU 48213 back order": 2,
    "sunny afternoon weather": 3,
    "car dealership weekend offer": 4,
    "oil discount coupon": 5,
    # ... (6 more, two per document; full list in corpus.py)
}
```

Running BM25 (Part 1), embeddings (Part 2), hybrid RRF (Part 3), and hybrid + reranking (Part 4) against all 18 queries:

| stage | recall@1 | recall@2 | recall@3 |
|---|---|---|---|
| BM25 alone | 0.89 (16/18) | 1.00 | 1.00 |
| Embeddings alone | 0.89 (16/18) | 1.00 | 1.00 |
| Hybrid RRF | **0.94 (17/18)** | 1.00 | 1.00 |
| Hybrid + reranked | **0.83 (15/18)** | 0.94 (17/18) | 1.00 |

The regression is still there — **hybrid fusion is still the best stage, reranking still makes it worse** — but it shrank from a 33-point gap (0.83 → 0.50 at n=6) to an 11-point gap (0.94 → 0.83 at n=18). That shrinkage is exactly the sample-size effect it looks like: checking per-query, **the same 3 queries break and the same 1 query gets fixed as before** — none of the 12 new, ordinary queries flipped under any stage. The absolute count of broken/fixed queries didn't change at all; only the denominator did.

| query | hybrid top-1 | reranked top-1 | effect |
|---|---|---|---|
| "car oil change" | doc0 (correct) | doc4 (wrong) | **reranking broke it** |
| "weekend weather" | doc4 (wrong) | doc3 (correct) | **reranking fixed it** |
| "oil change" | doc0 (correct) | doc5 (wrong, spam) | **reranking broke it** |
| "automobile SKU-48213-B service" | doc1 (correct) | doc2 (wrong) | **reranking broke it** |
| *(all 14 other queries)* | correct | correct | unchanged |

This is the answer to "could the original result just be small-sample noise": partly yes, in that the *magnitude* (33 points) overstated what a 6-query sample can responsibly claim. But no, in that the *mechanism* doesn't depend on sample size at all — these are the same three identifiable, explainable failures from Part 4 (doc4-over-doc0 preference inherited from Part 2's embeddings, doc5's keyword stuffing fooling the cross-encoder, doc2's exact-code trap undoing RRF's rescue of doc1), and adding 12 more ordinary queries didn't dilute them away, because they're not noise — they're reproducible properties of this specific cross-encoder on this specific kind of adversarial input. Net effect, now measured on 18 queries instead of 6: **-2 on recall@1**, same as before — the number got more honest, not smaller.

This doesn't mean cross-encoder reranking is bad in general — it's a standard, well-evidenced technique, and this corpus was deliberately stocked with adversarial cases built specifically to probe each stage's weak points across the whole series. What it does mean is that "add a reranker" is not automatically a win, and the only way to know whether it's a win *for your system, on your queries* is to measure it, exactly like this — on enough queries that the number means what it claims to mean. Trusting the standard advice without measuring it against your own ground truth would have silently shipped a regression.

---

## Beyond recall@k: precision@k, MRR, and NDCG

recall@k answers one question — is the correct document anywhere in the top-k? It says nothing about *where* in the top-k, which is what the next three metrics are for. Computed for real against the same four stages and the same ground truth, at k=1:

| stage | precision@1 | MRR | NDCG@1 |
|---|---|---|---|
| BM25 alone | 0.8889 | 0.9444 | 0.8889 |
| Embeddings alone | 0.8889 | 0.9444 | 0.8889 |
| Hybrid RRF | 0.9444 | 0.9722 | 0.9444 |
| Hybrid + reranked | 0.8333 | 0.9074 | 0.8333 |

Look closely and precision@1 and NDCG@1 are identical to recall@1 from the table above, row for row. That's not a coincidence worth glossing over — with this ground truth's single correct document per query, `precision@k` is an exact linear rescaling of `recall@k` (`precision@k = recall@k / k`, so at k=1 they're numerically identical), and NDCG collapses to the same binary signal as precision at k=1 with no graded relevance to show its actual strength (weighing "very relevant" against "somewhat relevant") off. Neither metric adds new information here; stating that plainly matters more than padding the post with metrics that look rigorous without actually measuring anything recall@k didn't already show.

**MRR is the one that does add something.** Unlike the others, it credits partial correctness when the right document is close but not first: hybrid RRF's MRR (0.9722) is higher than its precision@1 (0.9444), because the one query hybrid got "wrong" at rank 1 still had the correct document at rank 2, and MRR gives that partial credit while precision@1 and recall@1 give it nothing. Reranking's MRR (0.9074) tells the same regression story as its recall@1 (0.83), just on a continuous scale instead of a binary one — useful for seeing *how bad* a miss was, not just whether it happened.

---

## Groundedness: can a judge catch Part 5's fabrication?

Part 5 ended with a specific, concrete problem: the pipeline generated *"This suggests that the vehicle's engine oil and brake system require immediate attention or replacement as they have been out of stock for this period,"* correctly citing a real source document — which never mentions anything being out of stock. Citation attribution confirmed *which* source contributed vocabulary; it said nothing about whether the claim itself was true. This part tries to close that gap with an LLM-as-judge groundedness check, using the same small local model as Part 5 (`Qwen2.5-0.5B-Instruct`, no API key).

**Judge design 1 — a single holistic call:**

```
Source: {source}
Claim: {claim}
Is the claim fully supported by the source? Answer with exactly one word: SUPPORTED or UNSUPPORTED.
```

Tested directly against Part 5's real pipeline output (not a constructed example):

| claim | source | expected | judge said |
|---|---|---|---|
| "SKU-48213-B is an automotive component... on back order until the next week." | SKU doc | SUPPORTED | SUPPORTED ✓ |
| "...engine oil and brake system require immediate attention... **out of stock**..." | automobile/oil/brake doc | UNSUPPORTED | **SUPPORTED ✗** |

The naive judge fails on exactly the case it exists to catch. It correctly validates the genuinely grounded claim, then also validates the fabricated one — the same small model that generated the hallucination doesn't reliably catch its own category of mistake when asked to judge it directly, even in a completely fresh call with no memory of having generated it.

**Judge design 2 — force quoted evidence, don't ask for a holistic verdict:**

```
Source: {source}
Claim: {claim}
Quote the exact words from the source that support this claim. If no such support exists, respond with exactly: NO SUCH TEXT EXISTS.
```

This follows the same pattern that fixed abstention and citations in Part 5: don't trust a small model's holistic judgment, force it into a narrower, more mechanical task instead.

| claim | expected | judge said |
|---|---|---|
| "SKU-48213-B is an automotive component... on back order..." | SUPPORTED | **UNSUPPORTED ✗** ("NO SUCH TEXT EXISTS") |
| "...engine oil and brake system... out of stock..." | UNSUPPORTED | **UNSUPPORTED ✓** |
| "The SKU is currently unavailable and will not ship for several days." (a true claim, *paraphrased* rather than quoted) | SUPPORTED | **SUPPORTED ✓** |

Accuracy across this three-case set: naive judge 1/3, quote judge 2/3. The quote judge does catch Part 5's actual fabrication, which the naive judge missed entirely — real progress. But it now misses the *first* case, flagging a genuinely well-grounded claim as unsupported, apparently because that claim adds a few words beyond the source ("automotive component that needs to be replaced") and the model can't produce one exact quote covering the whole sentence, so it defaults to "no such text." Forcing evidence-pointing fixed a false negative (missed fabrication) at the cost of a new false positive (rejecting a true claim).

Neither judge is good enough to ship as the sole safeguard. The naive version is too permissive — it validates fabrications. The stricter version is too literal — it rejects true paraphrases. This is the same shape of tradeoff that's shown up repeatedly across this series (BM25 vs embeddings, RRF vs magnitude, reranking's own new failure modes): **tightening a check to catch one class of error reliably introduces a different class of error**, and a 500M-parameter local model doesn't have the headroom to hold both the narrow task and the general judgment together. Production groundedness checking typically leans on either a substantially larger judge model, or a purpose-built natural language inference (NLI) classifier trained specifically for entailment rather than a general chat model doing double duty — a different tool, not just a better prompt.

---

## What this means in practice

- **Measure, don't assume — and measure on enough queries that the number means what it claims to mean.** The recall@k table caught something query-by-query inspection in Part 4 had already shown, but the first pass of this measurement (6 queries) overstated the regression's *size* (33 points) even though it got the *direction* right. Growing the query set to 18 didn't change which queries failed or why — it just replaced an exaggerated number with an honest one (11 points). Both the original anecdotes and the inflated aggregate would have been true in direction and wrong in magnitude if left unchecked.
- **A judge model is a model, with its own failure modes, not an oracle.** Using an LLM to check another LLM's output doesn't sidestep the reliability problem, it just moves it one level up. The naive judge shares a blind spot with the generator it's checking.
- **Decomposing a task into a narrower, more mechanical one is a real technique, but it's not free.** It worked cleanly for abstention in Part 5 (relevance check: a true binary with no middle ground). It's messier for groundedness, because "is this specific wording supported" sits on a spectrum between verbatim and fabricated, and forcing a binary "quote it or admit nothing exists" answer collapses that spectrum in a way that creates new errors.
- **This is also a fitting place to end the core series.** Six parts ago, this started with raw term matching. It's ending with the sober conclusion that even the evaluation layer meant to catch mistakes has mistakes of its own — which is, honestly, the most production-realistic note a RAG series can end on.

---

## Where the series goes from here

This closes the six originally planned parts: lexical retrieval, semantic retrieval, hybrid fusion, reranking, generation, and evaluation — each one tested against the last, each fix verified rather than assumed.

The series kept going. [Part 7](/blog/rag-from-scratch-07-chunking/) measures chunking (chunk size, overlap, and strategy — none of which the 6-document corpus above was ever long enough to need). [Part 8](/blog/rag-from-scratch-08-vector-search-at-scale/) scales retrieval up to thousands of documents with an approximate nearest-neighbor index, plus metadata filtering. [Part 9](/blog/rag-from-scratch-09-multihop-retrieval/) tests query decomposition against questions that need two documents combined — with a genuinely surprising result in the opposite direction of this part's reranking finding. [Part 10](/blog/rag-from-scratch-10-long-context/) measures the actual cost of skipping retrieval entirely and stuffing a long-context model instead.

Two stretch topics from the original plan remain genuinely open, not covered by Parts 7-10: query rewriting/expansion (attacking the "INV-48213-B" bare-ID problem from the query side instead of the retrieval side), and a deeper dive into NLI-based groundedness checking now that this part has shown exactly why a general-purpose small LLM isn't quite enough for it.

If you want to pressure-test what you've picked up across this series, [RAG Interview Questions](/blog/rag-interview-questions/) covers the same material — easy to senior level — as question-and-answer, each answer backed by real numbers from these parts.

Full code for this post, and the rest of the series: [github.com/Thimmasani/rag-from-scratch](https://github.com/Thimmasani/rag-from-scratch) — see `retrieval/eval_retrieval.py`, `eval_generation.py`.
