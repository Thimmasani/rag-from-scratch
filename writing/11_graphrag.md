---
title: 'Building RAG From Scratch, Part 11: GraphRAG'
description: "Chunk-and-embed retrieval can't answer a question whose answer is a pattern across every document, not a sentence in any one of them. This part builds a small knowledge graph with local and global search, measured against a question no single chunk can answer."
pubDate: 'Nov 02 2026'
---

*Series: Building RAG From Scratch. Part 11. Every retrieval stage so far has answered "which document(s) are relevant to this query." This part asks a different kind of question: one whose answer is a pattern spread across every document, which no single retrieved chunk — however well-ranked — can contain on its own.*

---

"What's the most common root cause of incidents across all three teams?" has no single-document answer. Each incident report mentions one cause; the *answer* is a count across all nine of them. Every retriever built in this series so far — BM25, embeddings, hybrid, reranked — returns individual documents ranked by relevance to the query. None of them count anything. [GraphRAG](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/) (Microsoft Research) is built specifically for this gap: extract a knowledge graph from your documents, cluster it, pre-summarize the clusters, and answer broad questions by reasoning over the summaries instead of searching for the one best-matching chunk.

**One disclosure up front, same category as Part 8's `hnswlib` exception:** real GraphRAG clusters its knowledge graph using Leiden or Louvain community detection — genuine graph-partitioning algorithms, not something to hand-roll for a fixture this size. This part's "communities" are built by the much simpler rule of grouping documents that share a root-cause entity, which is what Leiden would converge toward on a graph this small and this cleanly separated anyway. Stated explicitly, not hidden.

---

## The fixture: nine incidents, one hidden pattern

`retrieval/graphrag_corpus.py`: nine short incident reports, each mentioning exactly one team (Atlas, Orion, or Nimbus), one system (checkout, payment, search, or notification), and one root cause:

```python
DOCUMENTS = [
    "Team Atlas's checkout service had an outage under high load. Root cause: "
    "a database connection pool timeout. Resolution: the pool size was increased.",
    "Team Orion's payment service had an outage during a traffic spike. Root "
    "cause: a database connection pool timeout. Resolution: read replicas were added.",
    # ... 7 more
]
```

By construction, "database connection pool timeout" is the root cause in 4 of the 9 incidents, spanning all three teams — the dominant pattern, buried across documents that otherwise look unrelated (different teams, different systems).

---

## 1. Entity extraction: 6 of 9, with a specific, repeatable failure

Step one of building the graph: extract (team, system, cause) from each report, using the local `Qwen2.5-0.5B-Instruct` model with a structured-output prompt (same pattern as Part 9's query decomposition):

```python
def extract_entities(document: str, generator: Generator) -> dict | None:
    prompt = f"""Extract the team name, system name, and root cause from this incident report.
Respond in EXACTLY this format, nothing else:
TEAM: <team name>
SYSTEM: <system name>
CAUSE: <root cause, a few words>

Incident report: {document}"""
    ...
```

Checked against the fixture's known ground truth: **6 of 9 correct.** The 3 misses share a specific pattern — the model swapped which field held which value. For doc1 ("Team Orion's payment service had an outage... Root cause: a database connection pool timeout"), it extracted `team: "Orion's Payment Service"` and `system: "Database Connection Pool Timeout"` — the cause landed in the system slot, and the team/system got merged into one string. The same swap happened on docs 4 and 7. All three misses are sentences where the system name sits right next to a possessive ("Orion's X service"), a different surface pattern from the cleaner "Team Atlas's checkout service" phrasing the model handled correctly every time. A 500M-parameter model extracting structured fields from free text is reliable on the common phrasing and not on its own minor variations — exactly the kind of thing that's invisible until you check against real ground truth, not just eyeball a couple of outputs.

## 2. Local search: a dead heat

For ordinary single-document lookups ("What caused Team Nimbus's search service slowdown?"), graph traversal (walk from the team node and the system node to their shared document) and flat embedding retrieval both got all 3 test queries right:

| query | graph traversal | flat embeddings |
|---|---|---|
| "What caused Team Atlas's checkout service outage?" | doc0 ✓ | doc0 ✓ |
| "What caused Team Nimbus's search service slowdown?" | doc5 ✓ | doc5 ✓ |
| "What caused Team Orion's notification service failure?" | doc7 ✓ | doc7 ✓ |

No daylight between them here, and that's an honest, expected result, not a letdown: local, single-entity lookups are exactly what plain retrieval already handles well. GraphRAG's local search mode exists for *this* case too, but it isn't where the technique earns its complexity — that's the next part.

## 3. Global search: the thing flat retrieval structurally cannot do

Indexing time produces pre-written community summaries (one line per root cause, not per document):

```
- Root cause 'database connection pool timeout': 4 incident(s), affecting teams Atlas, Nimbus, Orion.
- Root cause 'missing database index': 2 incident(s), affecting teams Atlas, Nimbus.
- Root cause 'payment gateway timeout': 1 incident(s), affecting teams Orion.
- Root cause 'misconfigured retry policy': 1 incident(s), affecting teams Atlas.
- Root cause 'message queue backlog': 1 incident(s), affecting teams Orion.
```

Asking the question against these summaries (not the raw documents):

> **GraphRAG answer:** "The most common root cause of incidents across all teams is 'database connection pool timeout.' This issue affects 4 incidents out of the total 6 incidents reported."

The headline numbers are right: correct cause, correct count (4). But the sentence also claims "6 incidents reported" — the actual total is 9. The model got the two numbers the question actually asked for correct, then added a third, unprompted, wrong number anyway. Even reasoning over clean, pre-aggregated summaries instead of raw noisy text, a small model can still fabricate a detail nobody asked about — the same lesson Part 5 and Part 6 already drew about this exact model, from a different angle.

Now the same question against flat embedding retrieval, no graph involved:

```
0.2779  doc7: Team Orion's notification service failed... Root cause: a message queue backlog...
0.2768  doc6: Team Atlas's payment service failed... Root cause: a misconfigured retry policy...
0.2708  doc3: Team Nimbus's notification service failed... Root cause: a database connection pool timeout...
```

Every one of these is a single incident. The top-ranked result (doc7) isn't even an instance of the actual dominant cause — it's the *least* common one, just the document whose wording happens to sit closest to the query's phrasing in embedding space. None of the three, individually or as a set, states that any cause recurs 4 times across 3 teams, because that count isn't a property of any one chunk — it's a property of the whole corpus, which a per-chunk similarity score was never going to surface.

---

## What this means in practice

- **Local vs. global isn't about which retrieval method is "better" — it's about which *kind* of question you're asking.** Flat retrieval tied graph traversal on local lookups and had no way to even approach the global one. GraphRAG's own local search mode doesn't claim an advantage here either; the value is entirely in the global case.
- **Entity extraction accuracy is a real, checkable number, and 6/9 is worth taking seriously before trusting a graph built on it.** A knowledge graph is only as good as the extraction step under it — this part's one failure pattern (field-swapping on a specific sentence structure) would silently mislabel a chunk of a bigger graph the same way, at a scale where nobody's checking every extraction by hand.
- **Pre-aggregating information doesn't make a small model immune to fabricating extra detail.** The global search answer got the two numbers that mattered right and invented a third one nobody asked for. Summarization reduces surface area for errors; it doesn't eliminate the model's tendency to add unsupported specifics when generating prose.
- **The gap this part closes is real and specific: synthesis across documents, not relevance within one.** Nothing in Parts 1-10 could have answered the global question correctly, not because any individual retriever was bad, but because the information those retrievers return — ranked individual chunks — was never going to contain an aggregate count in the first place.

---

## What's next

[Part 12](/blog/rag-from-scratch-12-vectorless-rag/) looks at a different alternative to chunk-and-embed retrieval: skipping vectors entirely and having the model navigate a document's own hierarchical structure instead, measured against the specific failure mode that produces — near-identical sentences repeated across different sections, where embedding similarity alone can't tell them apart.

Full code for this post: [github.com/Thimmasani/rag-from-scratch](https://github.com/Thimmasani/rag-from-scratch) — see `retrieval/graphrag_corpus.py`, `graphrag.py`, `eval_graphrag.py`.
