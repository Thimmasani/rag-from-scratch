---
title: 'Building RAG From Scratch, Part 10: RAG vs. Long-Context LLMs'
description: "With context windows now large enough to just stuff everything in, does retrieval still matter? Measured directly: token cost and latency for retrieving vs. stuffing, and whether a fact buried at different positions in a long context actually gets found."
pubDate: 'Oct 26 2026'
---

*Series: Building RAG From Scratch. Part 10, closing this round of additions. Modern LLMs support context windows in the hundreds of thousands to millions of tokens. If you can just paste your whole knowledge base into the prompt, why build nine parts of retrieval machinery at all? This part measures the actual cost of not retrieving, instead of answering the question in the abstract.*

---

This part reuses two existing fixtures rather than inventing a third, continuing a pattern already established in Part 5 (reusing Part 1's tokenizer): Part 7's `chunking_corpus.FACTS` supplies the "needle" — the one fact that actually answers the question — and Part 8's `generate_synthetic_corpus` supplies filler text to pad a context out to a controlled, verified token count using the real `Qwen2.5-0.5B-Instruct` tokenizer (not a word-count approximation like Part 7's chunking, since this part is specifically about LLM context windows).

---

## 1. RAG vs. stuffing: the cost table

Same query, same correct fact, two approaches: (a) retrieve just the one relevant chunk and generate from it, (b) skip retrieval and stuff a padded context containing the same fact plus filler, at three context sizes:

| target context size | RAG prompt tokens | RAG latency | Stuffing prompt tokens | Stuffing latency | latency ratio |
|---|---|---|---|---|---|
| 500 | ~56 | 0.80s | 500 | 3.96s | 4.9x |
| 2,000 | ~56 | 0.78s | 2,000 | 17.52s | 22.3x |
| 4,000 | ~56 | 0.80s | 4,000 | 47.99s | **59.8x** |

RAG's cost stays flat — it only ever sends the one retrieved chunk, regardless of how big the hypothetical full knowledge base is. Stuffing's cost scales directly with context size, and the *ratio* gets dramatically worse as context grows, not just the absolute latency: going from 500 to 4,000 tokens of filler is an 8x increase in context size but a 12x increase in the stuffing-vs-RAG latency ratio (4.9x → 59.8x), because generation cost isn't linear in prompt length on this model/hardware. Scale this to a real knowledge base — thousands of documents, not one padded toy context — and "just stuff it all in" stops being a simplification and starts being the thing that makes a system too slow and too expensive to run per query.

All three stuffed-context answers were still correct, for what it's worth — this table is about cost, not failure. The failure case is the next experiment.

## 2. Lost in the middle: did it actually show up?

A fixed 2,000-token context, the same needle fact inserted at the start, middle, or end, checking whether the generated answer actually contains the real fact ("45 days"):

| position | grounded | answer | contains "45 days"? |
|---|---|---|---|
| start | True | "A return must be initiated within 45 days of delivery to get a full refund." | **True** |
| middle | True | "To get a full refund when returning an item, it must be initiated within 45 days of delivery." | **True** |
| end | True | "When a return must be initiated within 45 days of delivery for a full refund." | **True** |

All three positions found the fact correctly. "Lost in the middle" — the documented tendency for models to attend less reliably to information buried in the middle of a long context — is a real, widely reported phenomenon, but it did not materialize on this specific test: this model, at this context length (2,000 tokens), with this fixture's filler text, retrieved the needle regardless of position. Reporting a clean "yes, the middle position lost it" result anyway, because that's the expected textbook finding, would be exactly the kind of unverified claim this series tries not to make — same honesty standard Part 8 applied when HNSW's recall/latency trade-off didn't show up on that fixture either. The likely reason here: 2,000 tokens and a 0.5B-parameter model with a 32K token context window is nowhere near the scale (typically tens of thousands of tokens, often on larger models with weaker long-range attention) where published "lost in the middle" results are usually measured. The honest conclusion is narrower than "this proves long context is safe": it means this specific small-context, small-model test wasn't the setup that exposes the effect, not that the effect doesn't exist.

---

## What this means in practice

- **The cost argument for RAG doesn't need "lost in the middle" to be true to hold.** Even with every stuffed-context answer coming back correct, the latency ratio alone (4.9x at 500 tokens, 59.8x at 4,000) makes "just stuff everything in" an increasingly bad default as the knowledge base grows, independent of whether accuracy degrades.
- **A negative result is still a result, and it's worth reporting as cleanly as a positive one.** This part set out to measure "lost in the middle" directly and didn't find it at the scale tested. That's a real, useful data point — it bounds the claim ("this effect needs more scale than 2K tokens on a 0.5B model to show up here") rather than confirming or refuting the general phenomenon.
- **Long context and retrieval solve different problems, and this part's own numbers show both sides of that.** Long context can genuinely find a needle regardless of position, at least at this scale — that's real capability, not a myth. It just doesn't make finding and sending the needle for free, which is the problem retrieval actually solves.
- **"Do I still need RAG" is a cost question as often as it's an accuracy question.** It's tempting to frame this as purely about whether long-context models can find the right information. The latency ratios in the first table say that even when they can, doing it by brute-force inclusion instead of retrieval gets disproportionately more expensive as the context grows — a production concern independent of correctness.

---

This closed out Parts 7-10 as originally scoped. The series kept going: [Part 11](/blog/rag-from-scratch-11-graphrag/) builds a small knowledge graph to answer questions whose answer is a pattern across every document, not a sentence in any one of them. [Part 12](/blog/rag-from-scratch-12-vectorless-rag/) skips vectors entirely and has the model navigate a document's own section structure instead.

Full code for this post: [github.com/Thimmasani/rag-from-scratch](https://github.com/Thimmasani/rag-from-scratch) — see `retrieval/longcontext.py`, `eval_longcontext.py`. For the original six-part core series (lexical retrieval through evaluation), start at [Part 1](/blog/rag-from-scratch-01-tfidf-bm25/). For a question-and-answer format covering this same material end to end, see [RAG Interview Questions](/blog/rag-interview-questions/).
