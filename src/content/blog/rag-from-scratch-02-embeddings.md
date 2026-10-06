---
title: 'Building RAG From Scratch, Part 2: Semantic Retrieval With Embeddings'
description: 'Swapping lexical matching for sentence embeddings on the same toy corpus from Part 1 — where embeddings fix vocabulary mismatch, where they route a bare ID to the wrong document entirely, and what that means for hybrid search.'
pubDate: 'Oct 04 2026'
---

*Series: Building RAG From Scratch. Part 2 of ~6. Part 1 covered lexical retrieval (TF-IDF, BM25). This part adds embeddings and runs them on the exact same corpus and queries, so every number is directly comparable.*

---

Part 1 ended with a clear limitation: TF-IDF and BM25 both failed on "weekend weather" the same way, and neither has any concept that "car" and "automobile" mean the same thing. Both are surface-token matchers, nothing more. Embeddings exist to fix exactly that gap.

This part swaps the scoring function, not the corpus. Same six documents and five queries from Part 1, plus one new document and one new experiment at the end. If you haven't read Part 1, the short version: doc0 and doc1 are the same topic (car maintenance) in different words, doc2 has a one-off exact code (`SKU-48213-B`), doc3 is an unrelated weather sentence, doc4 overlaps doc0's vocabulary, and doc5 is a keyword-stuffed spam document.

---

## What an embedding actually is here

An embedding model maps text to a dense vector where geometric closeness tracks semantic similarity, not lexical overlap. Two sentences with zero shared words can land close together if they mean similar things; two sentences sharing several words can land far apart if the words are used in unrelated senses.

For this part I used `sentence-transformers/all-MiniLM-L6-v2` — a small (384-dimensional), fast, widely used sentence embedding model. The retrieval logic is still from scratch: no vector-DB `.search()`, no LangChain `Retriever`, just embed and rank by a dot product computed directly.

```python
from sentence_transformers import SentenceTransformer

class EmbeddingRetriever:
    def __init__(self, documents: list[str], model_name: str = "all-MiniLM-L6-v2"):
        self.documents = documents
        self.model = SentenceTransformer(model_name)
        # normalize_embeddings=True -> unit-length vectors, so dot product == cosine similarity
        self.doc_embeddings = self.model.encode(documents, normalize_embeddings=True)

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        query_embedding = self.model.encode(query, normalize_embeddings=True)
        scores = self.doc_embeddings @ query_embedding
        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
        return [(idx, float(score)) for idx, score in ranked[:top_k]]
```

### Why normalize, then dot product instead of cosine directly

Cosine similarity is `(a·b) / (||a|| * ||b||)` — it isolates the *angle* between two vectors and ignores their length. That's exactly what you want for text: a short sentence and a long paragraph about the same topic shouldn't be judged "less similar" just because one vector happens to be longer.

If you L2-normalize both vectors first (scale each to length 1), the denominator becomes `1 * 1 = 1`, so cosine similarity reduces to a plain dot product. That's the whole reason `normalize_embeddings=True` is set above — it lets `self.doc_embeddings @ query_embedding` (a single matrix-vector multiply) give the same ranking as computing full cosine similarity per pair, with less code and faster execution. This is also precisely the normalization step that makes MIPS-based ANN indexes (FAISS, HNSW) usable for cosine-style semantic search, since those indexes are built to optimize dot product or Euclidean distance, not cosine directly.

---

## Running it on Part 1's queries

Here's the actual three-way comparison — TF-IDF, BM25, and embeddings — on the same four original queries plus the fifth (`"oil change"`) added in Part 1 for the keyword-stuffing example:

**Query: `"car oil change"`**

| doc | TF-IDF | BM25 | Embeddings |
|---|---|---|---|
| doc4: car/oil, overlaps doc0 | 0.4310 | 2.4034 | **0.6441** |
| doc0: car/oil/brakes, natural | 0.4530 | 2.4034 | 0.5903 |
| doc1: automobile/oil/brakes | 0.1057 | 0.5104 | 0.4871 |
| doc5: "oil" × 11, spam | 0.4870 | 1.3915 | 0.3865 |

Two things worth noticing. First, embeddings rank doc1 (which says "automobile," not "car") much closer to the top two than either lexical method does — 0.4871 vs TF-IDF's 0.1057. Second, embeddings also push the keyword-stuffed doc5 to last place, same conclusion BM25 reached but for a different reason: BM25 demotes it via explicit term-frequency saturation, embeddings demote it because repeating one word eleven times doesn't change what the sentence *means* very much, and the embedding reflects meaning, not term counts.

**Query: `"automobile brake service"`**

| doc | TF-IDF | BM25 | Embeddings |
|---|---|---|---|
| doc1: automobile/oil/brakes | 0.5641 | 4.1587 | **0.5647** |
| doc0: car/oil/brakes, natural | 0.1622 | 1.1432 | 0.4842 |
| doc4: car/oil, overlaps doc0 | 0.0000 | 0.0000 | 0.2609 |
| doc5: "oil" × 11, spam | 0.0519 | 0.7044 | 0.1507 |

This is the clearest vocabulary-mismatch result in the set. doc0 doesn't contain the word "automobile" anywhere, so both lexical methods barely register it (TF-IDF 0.1622) or, in doc4's case, score it a flat **0.0000** — zero shared terms means zero score, always, no matter how related the content actually is. Embeddings give doc0 a 0.4842 and even doc4 a nonzero 0.2609. That's not a rounding difference, it's a structural one: lexical scoring is mathematically incapable of expressing "these share no words but are still related," while embeddings do it by default.

**Query: `"weekend weather"`**

| doc | TF-IDF | BM25 | Embeddings |
|---|---|---|---|
| doc3: actually about weather | 0.2077 | 1.5868 | **0.3948** |
| doc4: car dealership, says "weekend" | 0.2486 | 1.7104 | 0.3379 |

Part 1 flagged this as lexical retrieval's most embarrassing failure: both TF-IDF and BM25 ranked doc4 (a car dealership ad that happens to contain the literal word "weekend") above doc3 (the document actually about weather), purely because "weekend" and "weather" both happened to be rare, single-document terms with similar IDF. Embeddings fix this cleanly — doc3 wins, 0.3948 to 0.3379 — because the model understands "weekend weather" as a semantic unit (a forecast-type query) rather than two independent token lookups.

**Query: `"oil change"`**

| doc | TF-IDF | BM25 | Embeddings |
|---|---|---|---|
| doc0: natural phrasing | 0.3368 | 1.2602 | **0.5791** |
| doc4: overlaps doc0 | 0.3205 | 1.2602 | 0.5547 |
| doc5: "oil" × 11, spam | **0.6551** | 1.3915 | 0.4853 |
| doc1: automobile/oil/brakes | 0.1422 | 0.5104 | 0.4559 |

Recall from Part 1: TF-IDF ranked the spam document doc5 **first** here (0.6551), nearly double doc0's score, purely from repeating "oil" eleven times. Embeddings don't make that mistake either — doc5 drops to third. Repetition without added meaning doesn't move an embedding much, since the model is encoding what the sentence is about, not counting words.

---

## Where embeddings actually break: the wrong-document case

Part 1's "SKU-48213-B" query was lexical retrieval's strongest case — rare exact string, both TF-IDF and BM25 nail it immediately. Embeddings get it right too:

| doc | Embeddings |
|---|---|
| doc2: exact SKU code | **0.5710** |
| doc0 | 0.1219 |

To find a real failure, I added a second code-bearing document to the corpus — doc6, an invoice number from a completely different domain (billing, not inventory), but one that happens to share digits with doc2's SKU code:

```python
# doc2: "replacement part SKU-48213-B is on back order until next week"
# doc6: "invoice number INV-77213-K was paid on March 3rd"
```

Now query a bare, invoice-style ID that reuses doc2's exact digits with a different prefix — `"INV-48213-B"`, which should obviously route to the invoice document, not the SKU one:

| query | top result | score | 2nd place | score |
|---|---|---|---|---|
| `"INV-48213-B"` (bare ID, wrong prefix convention) | **doc2 (SKU) — wrong** | 0.4217 | doc6 (invoice, correct answer) | 0.4092 |
| `"invoice INV-48213-B"` (+ one word of context) | doc6 (invoice) — correct | **0.7727** | doc2 (SKU) | 0.3853 |
| `"INV-48213-B was paid"` (+ more context) | doc6 (invoice) — correct | **0.6559** | doc2 (SKU) | 0.4139 |

This is the real failure, and it's a close one — 0.4217 vs 0.4092, basically a coin flip. The model is pattern-matching on the shared digits ("48213-B") rather than the prefix that actually distinguishes the two document types ("INV-" means invoice, "SKU-" means inventory part, but the model doesn't weight that distinction strongly from a bare code alone). The instant you add a single word of natural-language context — just "invoice" — the correct document jumps to 0.7727 and the wrong one drops to 0.3853. One word of context completely flips the ranking.

That's the actual, concrete lesson: embeddings are matching *meaning*, and a bare identifier barely has any — so the model falls back to whatever weak signal is left (shared digits, structural similarity), which can point at the wrong document when two different ID systems happen to overlap numerically. This is a realistic production scenario too: separate systems (inventory, billing, support tickets) frequently reuse number ranges, and users searching by raw ID without context are exactly the case embeddings handle worst.

---

## What embeddings cost you that lexical retrieval didn't

To be fair to Part 1, lexical retrieval had real advantages this part gives up:

- **No training, no model, no inference cost.** TF-IDF and BM25 are closed-form math over token counts — instant, deterministic, debuggable by hand. Embeddings require running a neural network for every document and every query.
- **Determinism and exact auditability.** You can hand-verify every TF-IDF/BM25 score (Part 1 did exactly that). Embedding scores come out of a 384-dimensional vector you can't meaningfully inspect term-by-term — you can see *that* two things are similar, not cleanly *why*.
- **No vocabulary drift risk.** A lexical index built today works identically forever. An embedding model has a version; swap models and your entire index's geometry changes, scores aren't comparable across model versions.
- **Guaranteed exact match.** Lexical methods guarantee a rare exact string ranks top whenever it's present and distinguish ID systems by their literal prefix. Embeddings can route a bare ID to the wrong document entirely when two systems overlap numerically, as shown above.

None of this makes embeddings worse, it makes them a different tool solving a different failure mode. Part 1's lexical methods are strong exactly where embeddings are weak (rare exact tokens, codes, IDs) and weak exactly where embeddings are strong (paraphrase, synonymy, "these mean the same thing but share no words").

---

## What's next

Both retrieval methods are now built and both have been shown to fail in specific, demonstrable ways on the same corpus. [Part 3](/blog/rag-from-scratch-03-hybrid-rrf/) combines them with Reciprocal Rank Fusion — and the result is more nuanced than "fusion always wins": it resolves some failures cleanly, but has its own honest blind spot when two retrievers disagree about which of two documents ranks first.

Full code for this post, and the rest of the series: [github.com/Thimmasani/rag-from-scratch](https://github.com/Thimmasani/rag-from-scratch) — see `retrieval/embeddings.py`, `compare_semantic.py` (produces the tables above). Uses `sentence-transformers`, installed in a local `.venv` per `retrieval/README.md`.
