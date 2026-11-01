---
title: 'Building RAG From Scratch, Part 7: Chunking'
description: "The toy corpus from Parts 1-6 is pre-chunked, which is exactly why it never had to deal with chunking. This part builds a longer, multi-sentence fixture and measures what chunk size, overlap, and chunking strategy actually do to recall@k."
pubDate: 'Oct 20 2026'
---

*Series: Building RAG From Scratch. Part 7. Parts 1-6 answered "how do you rank documents." This part and the next three answer a question those six parts never had to face, because their 6-document corpus was already chunked by construction: what do you do before retrieval even starts, when your source documents are too big to embed and search as single units?*

---

Every document in `corpus.py` is a single short sentence. That was deliberate — it kept Parts 1-6's hand-worked IDF tables and recall@k measurements reproducible — but it also means nothing in the series so far has ever had to split a document. Real documents are pages, articles, PDFs: too long to embed as one unit, and mostly irrelevant to any single query if you don't split them first. This part builds a small fixture long enough to chunk, and measures — not assumes — what chunk size, overlap, and chunking strategy actually do to retrieval.

---

## The fixture

Four synthetic documents (`retrieval/chunking_corpus.py`), each ~150-180 words, styled like product-FAQ sections (warranty, returns, account security, billing). Each document has exactly one specific, locatable fact sentence, placed at a different position in each document — near the start, in the middle, or near the end — so chunk boundaries interact with fact position differently across the set:

```python
DOCUMENTS = [
    "This section explains warranty coverage... The extended warranty "
    "covers accidental damage for up to 3 years from the purchase date. "
    "...",  # fact near the START
    "This section explains the returns and shipping policy... Returns "
    "must be initiated within 45 days of delivery for a full refund. "
    "...",  # fact in the MIDDLE
    "This section explains account security settings... Two-factor "
    "authentication can be enabled from the account settings page under "
    "the Security tab.",  # fact near the END
    "This section explains billing and subscription management... "
    "Subscription renewals are charged exactly 7 days before the current "
    "billing cycle ends. ...",  # fact in the MIDDLE
]
```

Ground truth here can't just be "which document is correct," the way `corpus.py`'s `GROUND_TRUTH` works for Parts 1-6. A document index doesn't tell you whether the retrieved *chunk* actually contains the fact, because chunk boundaries move every time chunk size or overlap changes — the same document gets split differently at every configuration. So the ground truth stores the fact's literal text instead, and correctness is checked fresh against whatever chunk a given configuration produces: `fact_text in chunk`.

---

## Two chunking strategies, both word-count based

```python
def fixed_size_chunks(text: str, chunk_size: int, overlap: int = 0) -> list[str]:
    if not (0 <= overlap < chunk_size):
        raise ValueError(f"overlap must satisfy 0 <= overlap < chunk_size, got overlap={overlap}, chunk_size={chunk_size}")
    words = text.split()
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks
```

A naive sliding window over words — it has no idea where a sentence starts or ends, which is exactly what makes it a useful baseline: it can and will cut a fact in half if the window happens to land there.

```python
def sentence_aware_chunks(text: str, target_size: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current, current_len = [], [], 0
    for sentence in sentences:
        sentence_len = len(sentence.split())
        if current and current_len + sentence_len > target_size:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sentence)
        current_len += sentence_len
    if current:
        chunks.append(" ".join(current))
    return chunks
```

Splits into sentences first (the same regex Part 5's citation attribution already uses), then greedily packs whole sentences up to a target size. A sentence never gets split, at the cost of variable chunk sizes.

Both are deliberately word-count based rather than LLM-token based — this series has no tokenizer dependency until Part 5's generation model, and chunk text needs to stay human-readable in the article. Production systems more commonly chunk by token count; it's a substitution of the unit, not a different mechanism.

---

## 1. A fact actually gets split

Take the returns document and its fact: `"Returns must be initiated within 45 days of delivery for a full refund."` — 13 words. At `chunk_size=13, overlap=0`:

```
[3] is approved through the account portal. Returns must be initiated within 45 days
[4] of delivery for a full refund. Items returned after the refund window may
```

The fact spans exactly one chunk boundary, with "within 45 days" ending chunk 3 and "of delivery for a full refund" starting chunk 4 — the fact is not intact in any single chunk. At the same `chunk_size=13` but `overlap=4`:

```
[5] Returns must be initiated within 45 days of delivery for a full refund.  <-- fact intact here
```

A 4-word overlap is enough to put the whole sentence inside one chunk window. This is the mechanical version of the interview article's chunking answer, made concrete: overlap isn't a vague "best practice," it's the specific thing that determines whether a fact sitting near a chunk boundary survives intact or gets cut in half.

---

## 2. Chunk size changes recall@k by a lot, not a little

Chunking every document at several fixed sizes (`overlap=0`), embedding the resulting chunks (Part 2's `EmbeddingRetriever`, unmodified — it takes any `list[str]`), and checking whether the fact text appears in the top-k retrieved chunks:

| chunk size (words) | total chunks | recall@1 | recall@3 |
|---|---|---|---|
| 15 | 38 | 0.25 (1/4) | 0.25 (1/4) |
| 30 | 20 | 0.25 (1/4) | 0.50 (2/4) |
| 60 | 11 | 0.75 (3/4) | 1.00 (4/4) |
| 120 | 7 | 1.00 (4/4) | 1.00 (4/4) |

At `chunk_size=15`, recall@1 is 0.25 — three of the four facts never surface as the top result at all, even allowing the top 3. At `chunk_size=120` (chunks close to full-document size for this fixture), recall@1 is a perfect 1.00. The reason isn't subtle: at 15 words, a chunk often contains the fact's surrounding filler sentence but not the fact itself, or contains only half the fact — and a half-fact or an off-topic filler sentence embeds as whatever it actually says, not as "the thing you were looking for." Smaller chunks mean more precise retrieval *when they contain the right content*, but only if the chunking boundary happens to land favorably. This fixture's naive fixed-size splitting mostly doesn't land favorably at small sizes.

This is the real-data version of the interview article's general claim that chunk size is "an empirical question for your specific corpus," not a fixed best practice — on this fixture, the empirical answer is "bigger, within the range tested," which is itself corpus-specific: a fixture with facts this short, surrounded by filler this generic, rewards wider windows that are more likely to contain a complete fact.

---

## 3. Sentence-aware chunking beats fixed-size at a matched average size

Fixed-size chunking's failures above are at least partly an artifact of not respecting sentence boundaries. Comparing the two strategies at the same *average* chunk size controls for that:

```
Sentence-aware (target_size=40): 18 chunks, avg length 29.3 words
Fixed-size (chunk_size=29, matched to that average): 20 chunks
```

| strategy | recall@1 | recall@3 |
|---|---|---|
| sentence-aware | **0.50 (2/4)** | **1.00 (4/4)** |
| fixed-size | 0.25 (1/4) | 0.50 (2/4) |

At essentially the same average chunk size, sentence-aware chunking doubles recall@1 and hits a perfect recall@3, where fixed-size chunking still misses half the facts even by the third result. The mechanism is exactly what you'd expect from never splitting a sentence: every fact in this fixture is one complete sentence, so a chunker that respects sentence boundaries can never cut one in half, while fixed-size chunking's word-count window has no such guarantee and sometimes does.

---

## What this means in practice

- **Chunk size is not a cosmetic parameter.** Recall@1 moved from 0.25 to 1.00 across the sizes tested here, on the same four facts, same embedding model, same retriever. That's a bigger effect than most of the retrieval-stage comparisons earlier in this series.
- **Overlap is a specific fix for a specific failure, not a vague safety margin.** It exists to keep a fact that lands near a chunk boundary from being split across two incomplete chunks — demonstrated directly above, not just asserted.
- **Respecting document structure beats a naive fixed window, at the same average chunk size.** Sentence-aware chunking's win here isn't about being bigger or smaller on average, it's about never truncating the one sentence that happens to matter.
- **This fixture's specific numbers are not universal constants.** "60-120 words beats 15-30 words" is true of *this* fixture's fact density and filler style, not a general rule. The actual, portable lesson is the one from Part 6: measure recall@k against your own content and your own queries, the same discipline applied here, rather than trusting a rule of thumb from a blog post (including this one).

---

## What's next

[Part 8](/blog/rag-from-scratch-08-vector-search-at-scale/) scales up: once a corpus has thousands of chunks instead of four documents, brute-force search over every embedding stops being "basically free," and metadata filtering (restricting search to a subset of documents by category, date, or permission) becomes its own real engineering problem.

Full code for this post: [github.com/Thimmasani/rag-from-scratch](https://github.com/Thimmasani/rag-from-scratch) — see `retrieval/chunking_corpus.py`, `chunking.py`, `eval_chunking.py`.
