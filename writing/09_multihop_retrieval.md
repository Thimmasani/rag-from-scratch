---
title: 'Building RAG From Scratch, Part 9: Multi-Hop Retrieval'
description: "Query decomposition is the standard fix for questions that need two documents combined. Tested directly against naive single-shot retrieval on the same fact-chain queries, it didn't win — a genuinely surprising result worth reporting exactly as measured."
pubDate: 'Oct 24 2026'
---

*Series: Building RAG From Scratch. Part 9. Every query in Parts 1-8 has one answer document (or, in Part 8's case, one ranked list of candidates). This part asks what happens when the real answer requires combining facts from two separate documents — and whether the standard fix for that (decomposing the question into sub-queries) actually helps on a case built to need it.*

---

Some questions can't be answered by matching the whole question to one document. "What deployment system does Project Helios's lead engineer's team use?" needs two separate facts: who the lead engineer is, and what system that specific person's team uses. Neither fact alone answers the question; a document mentioning the deployment system without the person link, or the person without the system link, isn't enough.

---

## The fixture: three fact chains, plus distractors

`retrieval/multihop_corpus.py`: three independent two-hop chains (10 documents total). Each chain is two short facts; around them, four distractor documents share vocabulary with one hop or the other without completing the chain:

```python
DOCUMENTS = [
    "Maria Chen is the lead engineer on Project Helios.",                          # hop1
    "Maria Chen's team relies on the Thornfield deployment system for releases.",  # hop2
    # ... two more chains (Orion/David Osei/Cascade, Nimbus/Priya Nair/Anchor) ...
    "Maria Chen also contributes to Project Orion's design reviews.",              # distractor: right person, wrong chain
    "The Thornfield deployment system was first adopted ... in 2024.",             # distractor: right system, no person link
    # ... two more distractors ...
]
```

Two retrieval strategies, both reusing existing series code unmodified:

```python
def single_shot_retrieve(query, hybrid, top_k):
    """The naive baseline: treat the whole multi-hop question as one
    query against Part 3's HybridRetriever."""
    return [idx for idx, _score in hybrid.search(query, top_k=top_k)]

def decompose_and_retrieve(query, hybrid, generator, top_k_per_hop):
    """Ask the generation model for a first-hop sub-query, retrieve with
    it; ask for a second-hop sub-query given the first, retrieve with
    that; return the union of both hops' candidates."""
    hop1_query = generator.generate_raw(hop1_prompt, max_new_tokens=30)
    hop1_results = hybrid.search(hop1_query, top_k=top_k_per_hop)
    hop2_query = generator.generate_raw(hop2_prompt, max_new_tokens=30)  # given hop1's query
    hop2_results = hybrid.search(hop2_query, top_k=top_k_per_hop)
    return union(hop1_results, hop2_results)
```

`generate_raw()` is a new public method on Part 5's `Generator` (added specifically so this reuse goes through a public method, the same convention `.answer()`/`.search()`/`.rerank()` already follow elsewhere in the series, rather than reaching into the underscore-prefixed `_generate` across a module boundary). If a generated sub-query doesn't look like a real question (doesn't end in `?`, or is suspiciously long), it falls back to the original full question rather than crashing.

---

## 1. Recall: decomposition didn't win

Checking whether *both* hop documents land in the retrieved set, across the three chains:

| query | single-shot (top-4) | both hops retrieved? | decomposed union | both hops retrieved? |
|---|---|---|---|---|
| Helios → Chen → Thornfield | [0, 1, 2, 7] | **True** | [0, 1, 2, 3] | **True** |
| Orion → Osei → Cascade | [2, 3, 6, 0] | **True** | [2, 0, 8, 9] | **False** |
| Nimbus → Nair → Anchor | [5, 4, 0, 2] | **True** | [4, 0, 2, 5, 8] | **True** |

**Single-shot: 3/3 chains with both hop documents retrieved. Decomposed: 2/3.**

This is the opposite of what the "decomposition helps multi-hop" framing predicts, and it's worth being specific about why. For the Orion/Osei/Cascade chain, the generated hop2 sub-query was *"What is the primary purpose of the monitoring platform used by the lead engineer's team at Project Orion?"* — a reasonable-sounding question that happens to retrieve doc9 (a distractor that only describes what the Cascade platform does generically) instead of doc3 (the actual hop2 fact, "David Osei's team relies on the Cascade monitoring platform"). The sub-query drifted from "which system does this specific person's team use" toward "what does this system do," a subtly different question that this fixture's distractor documents were specifically positioned to catch.

Meanwhile, single-shot retrieval over the full original question worked for all three chains, because this fixture is small (10 documents) and the full question already contains enough signal — both the person-finding vocabulary and the system-finding vocabulary — for hybrid retrieval to surface both hop documents directly, without needing to split the question apart at all.

## 2. End-to-end answer correctness: both approaches got the right final answer anyway

| query | single-shot answer | correct? | decomposed answer | correct? |
|---|---|---|---|---|
| Helios | "...uses the Thornfield deployment system." | ✓ | "...uses the Thornfield deployment system for releases." | ✓ |
| Orion | "...uses the Cascade monitoring platform for releases." | ✓ | "...uses the Cascade monitoring platform." | ✓ |
| Nimbus | "...Priya Nair's team uses the Anchor incident response tool for releases." | ✓ | "...uses the Anchor incident response tool." | ✓ |

Both approaches produced the correct composite answer on all three chains — including the Orion chain, where decomposition's retrieval missed the actual hop2 document (doc3). Looking at what context it ended up generating from instead: doc2 ("David Osei is the lead engineer on Project Orion") plus doc9 (the distractor describing what the Cascade platform generically does, with no person link) was apparently enough surface overlap for the model to assemble "David Osei's team uses Cascade" correctly anyway — a right answer stitched together from an incomplete, not-quite-grounded context, rather than from the actual fact that links the two. That's a different, quieter kind of near-miss than an outright wrong answer, and it wouldn't have been visible from the end-to-end answer check alone — only from checking retrieval recall separately, which is the entire reason this series checks both.

---

## What this means in practice

- **"Decompose the question" is not automatically better than asking it directly.** On this fixture, with a hybrid retriever already working from Parts 1-3, single-shot retrieval over the whole multi-hop question outperformed decomposition on recall. Decomposition adds a failure mode of its own — a generated sub-query that drifts semantically from what the original question actually needed — on top of whatever single-shot retrieval already risked.
- **Decomposition should earn its complexity on corpora and questions where single-shot retrieval demonstrably can't find both facts** — a much larger corpus with less vocabulary overlap between the full question and either hop's document, for instance — not applied by default because it sounds like the more sophisticated approach.
- **A correct final answer doesn't guarantee the chain was actually assembled from the right evidence.** The Orion case produced the right words from the wrong (incomplete) context. This is the multi-hop analogue of Part 5's citation-vs-verification gap: checking only the final answer's correctness can hide a retrieval failure that happened to not matter *this time*, which is exactly the kind of gap that does matter the next time the surface-level overlap isn't there to paper over it.
- **Measure recall and final-answer correctness separately, as two different questions.** This series has made this point before (Part 6), and this part is a second, independent confirmation of it: an end-to-end pass doesn't mean every stage behind it worked.

---

## What's next

[Part 10](/blog/rag-from-scratch-10-long-context/) asks a question this series hasn't addressed directly: with context windows now large enough to just stuff everything in, does retrieval even matter anymore? Measured directly — token cost, latency, and whether a fact buried in the middle of a long context actually gets used.

Full code for this post: [github.com/Thimmasani/rag-from-scratch](https://github.com/Thimmasani/rag-from-scratch) — see `retrieval/multihop_corpus.py`, `multihop.py`, `eval_multihop.py`.
