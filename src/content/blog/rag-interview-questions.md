---
title: 'RAG Interview Questions: Easy to Senior, With Real Data and Examples'
description: "RAG interview questions from fundamentals to system design, each answered with actual numbers and failure cases from a from-scratch RAG build — not generic textbook answers."
pubDate: 'Oct 14 2026'
---

Most RAG interview question lists give textbook answers: "BM25 is a lexical ranking function," "embeddings capture semantic meaning." True, but forgettable, and they don't hold up under a follow-up question. Every answer below instead points at a specific, reproducible result from the ["Building RAG From Scratch"](/blog/rag-from-scratch-01-tfidf-bm25/) series — real documents, real queries, real numbers, real code. If you want to verify any of these yourself, the series builds every component from scratch (no `sklearn`, no LangChain `Retriever`) and the code is at [github.com/Thimmasani/rag-from-scratch](https://github.com/Thimmasani/rag-from-scratch).

Organized by level: easy (fundamentals), mid (mechanisms and trade-offs), senior (system design and failure analysis).

---

## Easy

### Q1. What is RAG, and why use it instead of just fine-tuning a model on your data?

**Answer:** RAG (Retrieval-Augmented Generation) retrieves relevant documents at query time and feeds them into an LLM's context window, instead of baking knowledge into the model's weights. Two concrete reasons to prefer it over fine-tuning:

- **Knowledge updates don't require retraining.** Add, edit, or delete a document and the next query sees the change immediately. Fine-tuning requires a new training run every time the underlying facts change.
- **Answers can cite sources.** A RAG system can point at the specific document a claim came from; a fine-tuned model's output is an unverifiable blend of training data with no traceable origin.

The trade-off: RAG's answer quality is bounded by retrieval quality. If the right document isn't retrieved, no amount of generation skill fixes it — demonstrated directly in Part 6, where reranking regressed retrieval recall@1 from 0.83 to 0.50 on a 6-query eval set, and that regression flowed straight through to worse generated answers in Part 5's pipeline.

### Q2. What's the difference between TF-IDF and BM25?

**Answer:** Both score documents by term overlap with a query, weighted by how rare each term is (inverse document frequency). The difference is what happens as a term repeats:

- **TF-IDF** scales *linearly* with term frequency: a term appearing 10 times scores ~10x higher than appearing once.
- **BM25** *saturates*: `f * (k1+1) / (f + k1 * length_norm)` approaches a ceiling of `k1+1` as frequency grows, so going from 1 to 2 occurrences matters a lot, but 10 to 11 barely moves the score.

Concrete example from a toy corpus (one doc repeats "oil" 11 times with no real content, a second doc says it once naturally): on the query `"oil change"`, TF-IDF scored the spam doc 0.6551 vs the natural doc's 0.3368 — the spam doc nearly **2x** higher. BM25 scored them 1.3915 vs 1.2602 — barely a 10% gap. That's saturation visibly protecting against keyword stuffing. ([Part 1](/blog/rag-from-scratch-01-tfidf-bm25/))

### Q3. What is an embedding, and why can you compare two embeddings with a dot product?

**Answer:** An embedding is a dense vector representing a piece of text such that geometric closeness tracks semantic similarity — texts that mean similar things land near each other in vector space, even with zero shared vocabulary. If you L2-normalize two vectors to unit length first, their dot product equals their cosine similarity (the angle between them), which is why `normalize_embeddings=True` plus a plain `@` matrix multiply is a standard, fast way to rank documents by embedding similarity — no separate cosine computation needed. ([Part 2](/blog/rag-from-scratch-02-embeddings/))

### Q4. What's the difference between a bi-encoder and a cross-encoder?

**Answer:** A bi-encoder encodes the query and each document *independently* into fixed vectors, then compares vectors with a dot product. Because document embeddings don't depend on the query, they can be precomputed once and reused for every future search — this is what makes bi-encoders fast enough to search an entire corpus.

A cross-encoder feeds the query and *one* document into the model *together* as a single input and outputs a relevance score directly, with no precomputable vectors — every query-document pair needs a fresh forward pass. Measured directly on CPU: bi-encoder search time stays roughly flat (~5ms) regardless of candidate count, while cross-encoder reranking scales linearly — 1.1x slower at 10 candidates, 4.1x at 50, 16.2x at 200. That's why cross-encoders rerank a short candidate list instead of replacing retrieval. ([Part 4](/blog/rag-from-scratch-04-reranking/))

### Q5. What does recall@k mean, and why does it matter for retrieval specifically?

**Answer:** recall@k for a single query is 1 if the correct document appears anywhere in the top-k results, 0 otherwise, averaged across a query set. It answers "did we keep the right document in play at all," before worrying about exact rank order (that's precision/MRR/NDCG's job). It's the right first metric for retrieval because generation can't fix a document that retrieval never surfaced — if recall@k is low, improving your prompt or your LLM won't help. In a real measurement across four retrieval stages, recall@3 was a perfect 1.00 for every stage, but recall@1 varied from 0.50 to 0.83 — showing that "the right doc is somewhere in the results" and "the right doc is ranked first" are very different claims. ([Part 6](/blog/rag-from-scratch-06-evaluation/))

---

## Mid-level

### Q6. Why can't you just average BM25 scores and embedding cosine similarity to combine lexical and semantic retrieval?

**Answer:** The two scores live on incompatible scales. In a real measurement, BM25 scores on the same corpus ranged from 0.0 up to 4.1587 depending on term rarity and query length (unbounded above as a formula, even if this particular run topped out there), while cosine similarity is mathematically bounded in [-1, 1]. Averaging them directly means BM25 dominates every query purely because its numbers are bigger, not because it's more relevant. Min-max normalizing first just substitutes one arbitrary choice (which normalization scheme, over what reference set) for another, and that choice silently breaks whenever either retriever's score distribution shifts — a longer query, a swapped embedding model, a bigger corpus. The standard fix is Reciprocal Rank Fusion (see Q7), which sidesteps the scale problem by fusing rank positions instead of raw scores. ([Part 3](/blog/rag-from-scratch-03-hybrid-rrf/))

### Q7. How does Reciprocal Rank Fusion (RRF) work?

**Answer:** Each retriever runs independently and produces its own ranked list. A document's fused score is the sum of `1 / (k + rank)` across every list it appears in, where `rank` is its 1-indexed position in that list and `k` is a smoothing constant (commonly 60):

```
RRF(d) = Σ over each retriever r:  1 / (k + rank_r(d))
```

A document ranked #1 contributes `1/(k+1)`; ranked #10, a much smaller `1/(k+10)`. A document absent from one retriever's list simply contributes nothing from that retriever — no penalty beyond "didn't help." This requires no score normalization and no hand-tuned weighting, which is why it's a common default for combining lexical and semantic search.

**The catch, worth raising unprompted:** RRF discards magnitude entirely, only rank position matters. Two documents that are rank-1-in-one-list/rank-2-in-the-other get the *exact same* fused score regardless of how decisive either retriever's preference actually was. Measured directly: one retriever had a near-coin-flip preference (1.7104 vs 1.5868) for the wrong document, while the other had a decisive 17%-margin preference (0.3948 vs 0.3379) for the correct one — RRF fused these into an exact tie, `1/(61) + 1/(62)` on both sides, because it can't see the difference between "barely preferred" and "strongly preferred." ([Part 3](/blog/rag-from-scratch-03-hybrid-rrf/))

### Q8. Why is a cross-encoder not used to search an entire corpus directly?

**Answer:** It has nothing to precompute — every query-document pair requires its own full forward pass, since the model needs both texts present simultaneously to let its attention layers compare them. Measured directly: at 200 candidates, cross-encoder reranking took 69ms vs a bi-encoder's 4.28ms for the same comparison — 16x slower, and that gap grows linearly with corpus size since there's no caching possible. Run that over a 100,000-document corpus per query and the system becomes unusable. The standard pattern: use a fast method (BM25, embeddings, or RRF fusion) to narrow millions of documents to a short candidate list (10-100), then let the cross-encoder make the final call only on that short list. ([Part 4](/blog/rag-from-scratch-04-reranking/))

### Q9. What's the practical difference between retrieval and reranking in a RAG pipeline?

**Answer:** Retrieval is a recall-oriented stage: cast a wide net over the whole corpus cheaply, optimizing for "the right document is somewhere in the top-k." Reranking is a precision-oriented stage: take that short candidate list and reorder it using a more expensive, more accurate (in principle) model that looks at the query and each candidate jointly. The assumption going in is that reranking should only ever improve precision on top of retrieval's recall. That assumption doesn't always hold — see Q12 for a measured counterexample.

### Q10. Explain term-frequency saturation in BM25 and why `k1` exists.

**Answer:** BM25's term-frequency component is `f(t,d) * (k1+1) / (f(t,d) + k1 * length_norm)`, where `f(t,d)` is how many times term `t` appears in document `d`. As `f` grows, this expression asymptotically approaches `k1+1` — a ceiling. `k1` (typically 1.2-2.0) controls how quickly that ceiling is approached: a smaller `k1` saturates faster (repetition pays off less), a larger `k1` lets repetition matter longer before flattening out. This exists specifically to blunt keyword stuffing — a document repeating a word 11 times isn't 11x more relevant than one mentioning it once, and unsaturated scoring (like raw TF-IDF) rewards that repetition almost linearly, which is directly exploitable. ([Part 1](/blog/rag-from-scratch-01-tfidf-bm25/))

### Q11. Why might a RAG system fail to retrieve an exact product code or ID, even using embeddings?

**Answer:** Embeddings encode *meaning*, and an opaque identifier barely has any for the model to represent — it falls back to whatever weak signal is left. Measured directly: two documents from different domains (an inventory SKU and a billing invoice) happened to share digits. Querying the bare invoice-style ID `"INV-48213-B"` — which should obviously route to the invoice document — instead routed to the *wrong* document, the SKU one, at 0.4217 vs 0.4092, essentially a coin flip. The model was pattern-matching on the shared digits ("48213-B") rather than the prefix ("INV-" vs "SKU-") that actually distinguishes the two systems. Adding a single word of natural-language context ("invoice") flipped the ranking decisively back to correct — 0.7727 vs 0.3853. ([Part 2](/blog/rag-from-scratch-02-embeddings/))

The practical fix is hybrid retrieval (BM25 guarantees an exact string match when present) or format-aware query routing that detects ID-shaped tokens and handles them via exact lookup rather than embedding similarity.

---

## Senior

### Q12. You add a cross-encoder reranking stage to your RAG pipeline and recall@1 goes down, not up. How do you investigate, and what would you actually find?

**Answer:** This is a measured result, not a hypothetical: on a 6-query evaluation set, hybrid retrieval alone scored recall@1 = 0.83; adding reranking dropped it to 0.50. The investigation path:

1. **Don't trust the aggregate number alone — break it down per query.** Comparing hybrid-stage top-1 against reranked top-1 for every query revealed: reranking fixed exactly 1 case (a genuine tie the hybrid stage couldn't resolve) but broke 3 others that hybrid already had correct. Net: -2.
2. **Look at what the reranker is actually keying on in the broken cases.** Two of the three regressions traced to adversarial patterns the reranker was specifically vulnerable to: it preferred a document containing a literal exact-code string over the semantically correct document (same trap that fooled embeddings, but the cross-encoder was *more* confident in the wrong answer: an 11-point logit gap toward the wrong document), and it preferred a keyword-stuffed spam document over a natural, correctly-phrased one — a failure mode neither BM25 (via saturation) nor embeddings had.
3. **Conclude correctly: "add a reranker" is not a context-free win.** It's a standard, well-evidenced technique in general, but whether it helps *your* system depends on *your* query distribution, and the only way to know is to measure recall@k before and after on queries that actually probe your corpus's weak points — not just trust the technique's general reputation. ([Part 4](/blog/rag-from-scratch-04-reranking/), [Part 6](/blog/rag-from-scratch-06-evaluation/))

### Q13. Design a retrieval system that needs to handle both semantic search ("find documents about car maintenance") and exact-match lookups ("find the document mentioning SKU-48213-B") reliably.

**Answer:** Neither lexical nor semantic retrieval alone is sufficient, and naive hybrid fusion doesn't fully solve it either — this needs to be reasoned through with evidence, not asserted:

- **Pure BM25/lexical** nails exact codes (guaranteed top rank when an exact token match exists, since `df` for a rare token is low and IDF is correspondingly high) but fails on paraphrase — "automobile" when the query says "car" scores a flat 0.0 lexically, no matter how related the content is.
- **Pure embeddings** fix paraphrase but are unreliable on exact codes: can route to the *wrong* document entirely when two different ID systems happen to share digits (Q11).
- **RRF fusion of both** helps but has a specific, provable blind spot: it produces an exact mathematical tie whenever the two retrievers swap which of two documents ranks #1 vs #2, regardless of how different their underlying confidence actually was — so fusion alone doesn't reliably break the exact-code-vs-semantic-answer conflict either.
- **Cross-encoder reranking** on top can resolve some of those ties correctly, but is independently vulnerable to the exact same literal-overlap trap, sometimes more confidently than the bi-encoder.

The actually robust answer: **add explicit ID detection upstream of retrieval.** Detect ID-shaped query tokens (regex for code/SKU/invoice patterns) and route them through a deterministic exact-match lookup (a hash/index lookup against known ID fields) that bypasses embedding ambiguity entirely, falling back to hybrid semantic+lexical retrieval for everything else. This is a query-understanding fix, not a "pick a smarter retrieval model" fix — demonstrated across [Part 2](/blog/rag-from-scratch-02-embeddings/), [Part 3](/blog/rag-from-scratch-03-hybrid-rrf/), and [Part 4](/blog/rag-from-scratch-04-reranking/), the same failure mode survived three successive, increasingly sophisticated retrieval upgrades.

### Q14. How do you make a RAG system reliably say "I don't know" instead of hallucinating when retrieval doesn't find anything relevant? Why does the obvious approach fail?

**Answer:** The obvious approach — a single prompt instructing the model to "answer using only the context, and say you don't know if the context doesn't help" — is demonstrably unreliable, tested directly: given context about car maintenance and weather, and a question about the capital of France, a small instruction-tuned model answered *"The capital of France is Paris,"* then fabricated a justification connecting car maintenance and weather conditions to French geography, entirely outside the given context. The explicit abstention instruction simply didn't hold under the compound task of "judge relevance AND generate an answer AND produce an exact fallback string if needed," all in one generation.

**The fix: split it into two separate steps.** First, a dedicated relevance-check call asking only a binary question — "does this context contain information that answers this question, YES or NO" — nothing else. If NO, return a hard-coded Python string (`"I don't know based on the given context."`), never generated text. This reliably worked across repeated tests in both directions (correctly answering when context was relevant, correctly abstaining when it wasn't), because the model only ever has to hold one simple judgment at a time instead of several compound ones. This mirrors a broader principle: small models are far more reliable at narrow, mechanical sub-tasks than at compound judgment calls bundled into one instruction. ([Part 5](/blog/rag-from-scratch-05-generation/))

### Q15. How would you build a groundedness/hallucination check for a RAG system's output, and what are the real limitations of LLM-as-judge approaches?

**Answer:** Tested two designs directly against a real, previously-observed fabrication (a generated claim that cited a real source document that never actually supported the specific claim made):

- **Naive judge**: a single holistic prompt, "is this claim supported by this source? SUPPORTED or UNSUPPORTED." Result: it missed the fabrication entirely, scoring it SUPPORTED. 1/3 accuracy on a small labeled test set. The same small model that generated a hallucination doesn't reliably catch its own error category when asked to judge it in a fresh call.
- **Stricter judge**: force the model to quote the exact supporting text from the source, or say "NO SUCH TEXT EXISTS" if none exists — decompose the holistic call into a narrower, more mechanical one. Result: it correctly caught the fabrication this time, but introduced a *new* false positive, rejecting a genuinely true claim because it was paraphrased rather than quoted verbatim. 2/3 accuracy — real progress, but a different error class, not a clean fix.

**The limitation to state explicitly:** neither design was reliable enough to ship alone, and the gap isn't a prompting problem you can iterate your way out of with a ~500M-parameter model — tightening the check to catch fabrications made it too literal to accept true paraphrases. This points toward either a substantially larger judge model, or a purpose-built natural language inference (NLI) classifier trained specifically for entailment, rather than a general-purpose chat model doing double duty as both generator and judge. ([Part 6](/blog/rag-from-scratch-06-evaluation/))

### Q16. What's the difference between citation attribution and claim verification in a RAG system, and why does it matter?

**Answer:** Citation attribution answers "which source document contributed the words in this sentence" — it can be done cheaply and deterministically via lexical token overlap between the generated sentence and each candidate source, with no model call required. Claim verification answers a different, harder question: "is the specific factual claim in this sentence actually true according to that source."

The gap between them is real and demonstrated: a generation pipeline produced the sentence *"...the vehicle's engine oil and brake system require immediate attention or replacement as they have been out of stock for this period,"* correctly attributed to a document that does mention engine oil and brake service — but that document never says anything about being out of stock. The citation is technically correct (that document really did supply the vocabulary "engine oil and brake"); the specific claim ("out of stock") is fabricated. Attribution confirms *provenance*; it says nothing about *truth*. A system that only implements citation attribution (common, since it's cheap and doesn't need a model call) will confidently display fabricated claims with a real-looking, correctly-attributed citation next to them — arguably worse than no citation at all, since it manufactures false confidence. ([Part 5](/blog/rag-from-scratch-05-generation/))

### Q17. How do you decide between BM25, dense embeddings, and hybrid retrieval for a new RAG system?

**Answer:** This should be answered with the specific failure modes each one has, not a general preference:

- **Choose BM25-only** if your corpus is dominated by exact-match needs (legal citations, product codes, API error messages, ticket IDs) and paraphrase/synonymy is rare in your actual query logs. It's deterministic, needs no model, and scores are hand-auditable.
- **Choose embeddings-only** if your queries are dominated by natural-language paraphrase and synonymy (support questions, semantic search over prose) and your corpus has few or no exact identifiers that need guaranteed retrieval.
- **Choose hybrid (BM25 + embeddings via RRF)** as the default for most real-world RAG systems, since production query distributions are rarely purely one or the other — but go in aware of RRF's specific, provable blind spot (Q7): it can't distinguish "barely preferred" from "strongly preferred" when two retrievers disagree, so it won't cleanly resolve every conflict between the two signal types, only improve on having just one.
- **Add reranking only after measuring it, not by default** (Q12) — on at least one real measurement, it was a net negative on an adversarially-constructed query set, specifically because the cross-encoder shared and amplified a weakness (literal-overlap sensitivity) rather than compensating for it.

The meta-answer, which is itself a legitimate senior-level point: there's no universally correct choice, only a choice validated against your own query distribution and ground truth, measured the way [Part 6](/blog/rag-from-scratch-06-evaluation/) measured it — not assumed from general reputation.

---

Full code behind every number in this post: [github.com/Thimmasani/rag-from-scratch](https://github.com/Thimmasani/rag-from-scratch). The series this draws from starts at [Part 1: Lexical Retrieval](/blog/rag-from-scratch-01-tfidf-bm25/).
