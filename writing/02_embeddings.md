---
title: 'Building RAG From Scratch, Part 2: Semantic Retrieval With Embeddings'
description: 'Swapping lexical matching for sentence embeddings on the same toy corpus from Part 1 -- where embeddings fix vocabulary mismatch, where they degrade gracefully on exact codes, and what that means for hybrid search.'
pubDate: 'Oct 04 2026'
---

*Series: Building RAG From Scratch. Part 2 of ~6. Part 1 covered lexical retrieval (TF-IDF, BM25). This part adds embeddings and runs them on the exact same corpus and queries, so every number is directly comparable.*

---

Part 1 ended with a clear limitation: TF-IDF and BM25 both failed on "weekend weather" the same way, and neither has any concept that "car" and "automobile" mean the same thing. Both are surface-token matchers, nothing more. Embeddings exist to fix exactly that gap.

This part swaps the scoring function, not the corpus. Same six documents, same five queries from Part 1, plus one new experiment at the end. If you haven't read Part 1, the short version: doc0 and doc1 are the same topic (car maintenance) in different words, doc2 has a one-off exact code (`SKU-48213-B`), doc3 is an unrelated weather sentence, doc4 overlaps doc0's vocabulary, and doc5 is a keyword-stuffed spam document.

---

## What an embedding actually is here

An embedding model maps text to a dense vector where geometric closeness tracks semantic similarity, not lexical overlap. Two sentences with zero shared words can land close together if they mean similar things; two sentences sharing several words can land far apart if the words are used in unrelated senses.

For this part I used `sentence-transformers/all-MiniLM-L6-v2` — a small (384-dimensional), fast, widely used sentence embedding model. Worth being precise about what's "from scratch" here and what isn't: training a competitive embedding model yourself is a different, much larger project (that's what B3's dual-encoder training setup covers). What *is* from scratch in this post is the retrieval logic itself — we're not calling a vector-DB's `.search()` method or a LangChain `Retriever`. We embed, we rank by a dot product we compute ourselves, and that's it.

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

## Where embeddings get interesting: the exact-code query

Part 1's "SKU-48213-B" query was lexical retrieval's strongest case — rare exact string, both TF-IDF and BM25 nail it immediately. The natural assumption going into this part was that embeddings would fail here, since a model has no learned concept of an arbitrary product code.

Running it directly:

| doc | Embeddings |
|---|---|
| doc2: exact SKU code | **0.5710** |
| doc0 | 0.1219 |

That's not a failure — the embedding model actually ranks the right document first with a healthy margin. Worth being honest about this rather than forcing the "embeddings can't do exact match" narrative the setup implied. So I pushed further: is it actually understanding the code, or picking up on something else? I tried a second, structurally similar code the model has never seen in this corpus, both bare and wrapped in context:

| query | top result | score |
|---|---|---|
| `"SKU-48213-B"` (the real code, in doc2) | doc2 | **0.5710** |
| `"XJQ-99281-Z"` (unseen code, alone) | doc2 | 0.1845 |
| `"part number XJQ-99281-Z is on back order"` (unseen code + context) | doc2 | 0.5273 |

This is the actual finding, and it's more precise than "embeddings fail on codes." A bare, never-seen code scores weakly (0.1845 — barely above noise, and only still-highest because doc2 is structurally the only "code-like" document in a six-document corpus). But the same unseen code embedded in natural surrounding language ("part number ... is on back order") jumps straight back up to 0.5273, almost matching the real code's score. The model isn't matching the code string at all in that case — it's matching the surrounding words ("part number," "back order") against doc2's own phrasing. The code itself is just along for the ride.

That's an important distinction for RAG in production: embeddings don't hard-fail on identifiers, they degrade gracefully by falling back to whatever natural-language context surrounds the identifier. If your users search bare codes with no context ("SKU-48213-B" and nothing else), you're relying on coincidence, not retrieval. If they always type codes inside a sentence, embeddings alone might limp by. Either way, this is exactly the gap a lexical signal closes deterministically — which is the whole argument for hybrid retrieval in Part 3.

---

## What embeddings cost you that lexical retrieval didn't

To be fair to Part 1, lexical retrieval had real advantages this part gives up:

- **No training, no model, no inference cost.** TF-IDF and BM25 are closed-form math over token counts — instant, deterministic, debuggable by hand. Embeddings require running a neural network for every document and every query.
- **Determinism and exact auditability.** You can hand-verify every TF-IDF/BM25 score (Part 1 did exactly that). Embedding scores come out of a 384-dimensional vector you can't meaningfully inspect term-by-term — you can see *that* two things are similar, not cleanly *why*.
- **No vocabulary drift risk.** A lexical index built today works identically forever. An embedding model has a version; swap models and your entire index's geometry changes, scores aren't comparable across model versions.
- **Guaranteed exact match.** As shown above, lexical methods guarantee a rare exact string ranks top whenever it's present. Embeddings only approximate this, and the approximation's quality depends on context you don't control.

None of this makes embeddings worse, it makes them a different tool solving a different failure mode. Part 1's lexical methods are strong exactly where embeddings are weak (rare exact tokens, codes, IDs) and weak exactly where embeddings are strong (paraphrase, synonymy, "these mean the same thing but share no words").

---

## What's next

Both retrieval methods are now built and both have been shown to fail in specific, demonstrable ways on the same six-document corpus. Part 3 combines them: BM25 and embeddings run independently over the same corpus, their rankings get merged with Reciprocal Rank Fusion (RRF), and the fused ranking should beat either method alone across every query in this set, including the exact-code case this part just complicated.

Full code for this post: `ai_projects/rag_from_scratch/retrieval/` — `embeddings.py`, `compare_semantic.py` (produces the tables above). Uses `sentence-transformers`, installed in a local `.venv` alongside the pure-Python code from Part 1.
