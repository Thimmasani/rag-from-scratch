---
title: 'Building RAG From Scratch, Part 12: Vectorless RAG'
description: "No embeddings, no chunking, no vector index -- the model navigates a document's own section structure like a table of contents. Measured against flat embedding retrieval on a document deliberately built to confuse similarity search: three sections with nearly identical sentences."
pubDate: 'Nov 02 2026'
---

*Series: Building RAG From Scratch. Part 12, closing this second round of additions. Part 2 built retrieval on the premise that meaning lives in a vector. This part tests an alternative premise: meaning lives in a document's structure, and an LLM can navigate that structure directly, the way a person flips to the right chapter instead of fuzzy-matching sentences.*

---

"Vectorless RAG" (also called tree-navigation or PageIndex-style retrieval) skips embeddings, chunking, and vector search entirely. Instead: parse the document into its own hierarchy (sections, subsections, leaves), and at query time have the model read the *titles* at each level and pick which one to descend into — repeating until it reaches a leaf. No similarity score anywhere in the process.

---

## A fixture built to break embedding similarity on purpose

`retrieval/vectorless_corpus.py`: a service agreement with 3 plans (Residential, Business, Enterprise), each with the *same* 3 subsections (Pricing, Fees, Cancellation Terms) — deliberately near-identical in wording across plans, differing mainly in the numbers:

```
1. Residential Plan
  1.2 Fees: "...early termination fee of $50 if cancelled before the end of the first 3 months."
2. Business Plan
  2.2 Fees: "...early termination fee of $250 if cancelled before the end of the first 6 months."
3. Enterprise Plan
  3.2 Fees: "...early termination fee of $2,000 if cancelled before the end of the first 12 months."
```

"What is the early termination fee for the Business plan?" needs the model to know it's in the Business plan's section — the sentence structure is nearly identical across all three, so the number is almost the entire signal distinguishing them.

The document is represented exactly as that outline suggests — a plain nested Python dict, section titles and child lists, nothing else:

```python
DOCUMENT_TREE = {
    "title": "Service Agreement",
    "children": [
        {
            "title": "1. Residential Plan",
            "children": [
                {"title": "1.1 Pricing", "text": "The Residential Plan costs $19.99 per month..."},
                {"title": "1.2 Fees", "text": "The Residential Plan has an early termination fee of $50..."},
                {"title": "1.3 Cancellation Terms", "text": "Residential Plan cancellations take effect..."},
            ],
        },
        # ... Business Plan, Enterprise Plan, same shape
    ],
}
```

---

## How navigation actually works

```python
_LETTERS = "ABCDEFGHIJ"

def _choose_child(query: str, node: dict, generator: Generator) -> tuple[dict, bool]:
    """Asks the model to pick one of this node's child titles; falls back
    to the first child if the answer doesn't clearly match any title."""
    children = node["children"]
    titles = [c["title"] for c in children]
    lettered = "\n".join(f"{_LETTERS[i]}. {t}" for i, t in enumerate(titles))
    prompt = f"""Question: {query}

Which of these sections is most likely to contain the answer? Respond with
ONLY the letter of the section, nothing else.

Sections:
{lettered}"""
    response = generator.generate_raw(prompt, max_new_tokens=5).strip().upper()

    for i, title in enumerate(titles):
        if _LETTERS[i] in response:
            return children[i], False
    for child in children:  # fallback: substring match on title text
        if child["title"].lower() in response.lower():
            return child, False
    return children[0], True  # fallback: no match at all


def navigate_tree(query: str, generator: Generator, root: dict = DOCUMENT_TREE) -> tuple[str, str, bool]:
    """Descends top-down. Returns (section_path, leaf_text, any_fallback_used)."""
    node, path, any_fallback = root, root["title"], False
    while "children" in node:
        node, used_fallback = _choose_child(query, node, generator)
        any_fallback = any_fallback or used_fallback
        path = f"{path} > {node['title']}"
    return path, node["text"], any_fallback
```

Three levels in this tree (Service Agreement → Plan → Subsection), so `navigate_tree` makes three model calls per query, each one choosing among 2-3 titles — never looking at a single leaf's actual text until the final choice is made.

---

## A real bug, caught before it became a fake finding

The first version of tree navigation numbered the choice prompt `1. / 2. / 3.` — colliding with this fixture's section titles, which *already* start with `1. / 2. / 3.` ("1. Residential Plan"). The combined prompt effectively read "1. 1. Residential Plan / 2. 2. Business Plan / 3. 3. Enterprise Plan." Tested directly, the model answered "1" to every single query, regardless of content — 0 out of 6 correct, confirmed by checking the raw model output in isolation, not just the pipeline's final answer. That looked like a dramatic, clean "vectorless RAG completely fails" result. It wasn't a finding about the technique — it was a presentation bug. Switching the choice labels to letters (A/B/C) instead of numbers, with everything else unchanged, made the same model's answer correctly track the question every time it was tested in isolation.

This is worth stating plainly rather than quietly fixing and moving on: **a result that fails identically on every single input is a reason to suspect the harness before trusting the finding**, not a reason to publish a dramatic failure. The actual comparison below uses the corrected prompt.

---

## Tree navigation vs. flat embedding retrieval, 6 queries

Both approaches get the same 6 queries, over the same document — tree navigation reading section titles top-down, flat retrieval (Part 2's `EmbeddingRetriever`, unmodified) searching all 9 leaves by embedding similarity:

| query | tree navigation | flat embeddings |
|---|---|---|
| "early termination fee, Business plan?" | Business &gt; Fees ✓ | Business &gt; Fees ✓ |
| "early termination fee, Residential plan?" | Residential &gt; Fees ✓ | Residential &gt; Fees ✓ |
| "early termination fee, Enterprise plan?" | Enterprise &gt; Fees ✓ | Enterprise &gt; Fees ✓ |
| "Business plan, cost per month?" | Business &gt; Pricing ✓ | Business &gt; Pricing ✓ |
| "unused credits if I cancel Business plan?" | Business &gt; Cancellation ✓ | Business &gt; Cancellation ✓ |
| "refund if I cancel Enterprise plan?" | **Enterprise &gt; Cancellation ✓** | **Business &gt; Cancellation ✗** |

**Tree navigation: 6/6. Flat embeddings: 5/6.** The one query flat retrieval missed is exactly the case this fixture was built to expose: *"What refund do I get if I cancel the Enterprise plan?"* matched the *Business* plan's cancellation clause (score 0.7481) instead of the Enterprise plan's own (which never even appears in the top result). Both clauses open with almost the same sentence — "cancellations take effect at the end of the current billing month, with a ... refund for..." — and the embedding model weighted that shared structure over the one word (full vs. prorated) and section identity that actually mattered. Tree navigation never had this problem, because it never compared sentence similarity at all — it read "3. Enterprise Plan" at the top level and committed to that branch before the near-duplicate sentences downstream were even in view.

---

## What this means in practice

- **6 queries is a small sample — treat the margin, not just the direction, with appropriate caution.** One query's difference (6/6 vs 5/6) is a real, mechanistically explained win, not an overwhelming one; this series' own Part 6 addition is the reminder that a result's *direction* can be real while its *magnitude* needs more data before it's trustworthy at scale.
- **The one case tree navigation won is the specific, predicted case, not a random one.** Near-duplicate phrasing across different structural contexts is a known, named weakness of embedding similarity — this fixture was built to test for exactly that, and the single failure landed exactly where predicted. That's a more informative result than a bigger but less explainable margin would have been.
- **This technique trades a cheap, cached lookup for an LLM call at every level of the tree, at query time.** Flat embedding retrieval does one encode and one matrix multiply, however deep the corpus. Tree navigation here took 3 LLM calls per query (one per tree level) — fine for a document this small, a real cost that scales with tree depth, not corpus size, once documents get larger.
- **The debugging story matters as much as the result.** A result that's wrong on literally every input, with zero variation by query content, is a sign to check the harness before writing it up as a limitation of the technique — the number "0/6" would have been technically true and substantively misleading.

---

This closes this round of additions (Parts 11-12) to the "Building RAG From Scratch" series — GraphRAG for synthesizing a pattern across documents, vectorless tree navigation for retrieval that uses document structure instead of embedding similarity. Both are genuine alternatives to the chunk-and-embed approach Parts 1-10 were built on, each solving a specific, different limitation of it, and each with limitations of its own, measured rather than assumed.

Full code for this post, and the whole series: [github.com/Thimmasani/rag-from-scratch](https://github.com/Thimmasani/rag-from-scratch) — see `retrieval/vectorless_corpus.py`, `vectorless.py`, `eval_vectorless.py`. For the original series, start at [Part 1](/blog/rag-from-scratch-01-tfidf-bm25/).
