---
title: 'Building RAG From Scratch, Part 5: The Generation Layer'
description: 'Wiring retrieved and reranked context into actual LLM generation — why a single-shot "answer or abstain" instruction fails, why citation markers need to be attributed programmatically rather than generated, and what still goes wrong even after fixing both.'
pubDate: 'Oct 10 2026'
---

*Series: Building RAG From Scratch. Part 5 of ~6. Parts 1-4 built the full retrieval stack: lexical search, embeddings, hybrid fusion, reranking. This part is the "G" in RAG — taking retrieved context and actually generating an answer.*

---

Every part so far has been about ranking documents. This part is about what happens after you have them: construct a prompt, generate an answer, decide whether to trust it. The obvious approach — put the retrieved text in the prompt, ask the model to answer using only that context, and tell it to say "I don't know" if the context doesn't help — is also, demonstrably, not reliable enough to ship. Two specific failures showed up immediately when I actually tested it, not hypothetically.

Model for this part: `Qwen2.5-0.5B-Instruct`, a small (~500M parameter, ~1GB) open instruction-tuned model, run locally on CPU with `transformers` — no API key, consistent with how Parts 2 and 4 used local open models rather than a hosted API.

---

## Failure 1: single-shot abstention doesn't work

The natural first attempt:

```
Answer the question using ONLY the context below. If the context does not
contain the answer, say "I don't know based on the given context."

Context: [1] the car needs an oil change and new brake pads
         [2] the weather today is sunny with a light breeze in the afternoon

Question: What is the capital of France?

Answer:
```

Neither context chunk has anything to do with France. The instruction explicitly tells the model what to say if that's the case. The actual output:

> "The capital of France is Paris. According to the information provided in the context, the car needs an oil change and new brake pads, which suggests that these items are related to vehicle maintenance or safety. The fact that it's sunny and there's a light breeze in the afternoon implies that the weather conditions are suitable for driving, further supporting the idea that Paris is the capital city of France."

This is a genuinely striking failure, reproduced consistently across runs: the model answers correctly (Paris *is* the capital of France) but entirely outside the given context, then fabricates a tortured rationalization connecting car maintenance and weather to French geography, as if trying to retroactively justify an answer it had already decided to give. The explicit abstention instruction simply didn't take.

### The fix: make abstention a separate step, not a generation instruction

Splitting this into two steps — ask a yes/no relevance question first, and only generate an answer if the answer is yes — fixed this completely:

```python
def _is_relevant(self, query: str, numbered_context: str) -> bool:
    prompt = f"""Context:
{numbered_context}

Question: {query}

Does the context above contain information that directly answers the question? Respond with exactly one word: YES or NO."""
    response = self._generate(prompt, max_new_tokens=5)
    return response.upper().startswith("YES")
```

Tested on both directions:

| query | context | relevance check | result |
|---|---|---|---|
| "What does the car need?" | car maintenance doc | YES | generates grounded answer |
| "What is the capital of France?" | car + weather docs | NO | **fixed Python string returned, not generated** |

The critical design choice is the last part: when the check returns NO, the actual text shown to the user — `"I don't know based on the given context."` — is a hard-coded Python string, not something the model generates. The model only ever has to make a much simpler binary judgment (does this context address this question, yes or no), rather than the compound task of judging relevance *and* reliably producing an exact abstention sentence *and* not leaking outside knowledge, all in one generation. Separating concerns the model can't reliably hold together in one shot turned an unreliable instruction into two reliable ones.

---

## Failure 2: citation markers get silently dropped

Same pattern, different instruction. Asking the model to cite its sources inline, explicitly, with an example format:

```
Answer the question using ONLY the numbered context below. Cite sources
inline like [1].

Context: [1] the car needs an oil change and new brake pads
         [2] replacement part SKU-48213-B is on back order until next week

Question: What car maintenance is needed and is there a SKU on back order?

Answer:
```

Output:

> "The car maintenance required is an oil change and new brake pads. There is a SKU of 48213-B on back order until next week."

The answer is accurate and well-grounded in both sources. The citation markers — `[1]`, `[2]` — are just gone. Not malformed, not misplaced, simply never produced, consistently across repeated attempts and even after adding the relevance-check step from the previous fix.

### The fix: attribute citations programmatically, don't generate them

Rather than keep prompting harder, citations get attributed *after* generation, by checking token overlap between each generated sentence and each context chunk — reusing Part 1's `tokenize()` function, since this is the exact same term-overlap idea BM25 and TF-IDF are built on, just applied to attribution instead of ranking:

```python
def _attribute_citations(answer_text, context_chunks, min_overlap):
    sentences = re.split(r"(?<=[.!?])\s+", answer_text.strip())
    chunk_token_sets = [_content_tokens(chunk) for chunk in context_chunks]

    results = []
    for sentence in sentences:
        sentence_tokens = _content_tokens(sentence)
        cited = [
            i for i, chunk_tokens in enumerate(chunk_token_sets)
            if len(sentence_tokens & chunk_tokens) >= min_overlap
        ]
        results.append((sentence, cited))
    return results
```

Run on the uncited answer above, split sentence by sentence:

| sentence | cited source |
|---|---|
| "The car maintenance required is an oil change and new brake pads." | [1] |
| "There is a SKU of 48213-B on back order until next week." | [2] |

Clean, deterministic, correct, and it doesn't depend on the generation model's instruction-following at all.

### A real bug this caught: stopwords inflate overlap counts

Testing this against the "weekend weather" case from earlier parts surfaced an actual bug, not a model-behavior finding this time — a flaw in the citation design itself. The generated sentence "The weekend weather will be sunny with a light breeze in the afternoon" was attributed to *two* sources: the correct weather document, and also the unrelated car-dealership document (which only mentions "weekend" in passing). The overlap threshold was being met by two tokens — `"weekend"` and **`"the"`** — and `"the"` is a stopword that Part 1's `tokenize()` deliberately keeps (because Part 1's hand-worked IDF table explicitly lists "the" as a term, so that function can't change without breaking Part 1's hand-verified numbers).

The fix is a stopword filter scoped specifically to citation attribution, not a change to the shared tokenizer:

```python
_STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
              "has", "he", "in", "is", "it", "its", "of", "on", "or", "that",
              "the", "this", "to", "was", "were", "will", "with"}

def _content_tokens(text: str) -> set[str]:
    return set(tokenize(text)) - _STOPWORDS
```

After this fix, "weekend weather" correctly cites only the weather document. This is a small thing, but a useful reminder that even a deliberately simple, "from scratch" component has its own correctness edge cases worth testing for, not just the LLM's behavior.

---

## Running the full pipeline, start to finish

With both fixes in place, here's every query from the series run through the complete pipeline — BM25 + embeddings (Parts 1-2) → RRF fusion (Part 3) → cross-encoder reranking (Part 4) → generation with abstention and citations (Part 5):

**"car oil change"** → reranked to [doc4, doc0] → *"The car oil change is offered at the car dealership on weekends."* — cites doc4 and doc0, correctly grounded.

**"automobile brake service"** → reranked to [doc1, doc0] → *"The automobile brake service required is: new brake pads"* — correctly grounded, cites both.

**"weekend weather"** → reranked to [doc3, doc4] → *"The weekend weather will be sunny with a light breeze in the afternoon."* — cites only doc3 (the stopword fix from above, confirmed working end to end).

**"What is the capital of France?"** → reranked to [doc3, doc2], neither relevant → **abstains**: *"I don't know based on the given context."* Correct, despite the retrieval stage having nothing good to offer it — the pipeline doesn't force an answer just because retrieval returned *something*.

Two of the remaining results are worth slowing down on, because they're not clean wins.

### Citation isn't verification

**"SKU-48213-B"** reranks to [doc2, doc1] (consistent with Part 4's finding that this reranker favors the exact code string). The generated answer:

> "SKU-48213-B is an automotive component that needs to be replaced due to being on back order until the next week. [1] This suggests that the vehicle's engine oil and brake system require immediate attention or replacement **as they have been out of stock for this period.** [2]"

The first sentence is accurate and correctly cited to doc2. The second sentence cites doc1 correctly too — doc1 does mention engine oil and brake service — but doc1 never says anything about being "out of stock." That claim is a fabricated bridge between doc2's genuinely-out-of-stock SKU and doc1's unrelated maintenance note. The citation attribution is technically correct (doc1 really did contribute the vocabulary "engine oil and brake" to this sentence), but correct attribution doesn't mean the *claim* itself is actually supported by that source. Citing a source and verifying a claim against that source are two different problems, and this pipeline only solves the first one.

### A ranking failure becomes a useless answer

**"oil change"** reranks to [doc5, doc4] — Part 4 already showed this reranker favors the keyword-stuffed spam document over the natural one for this exact query. The generated answer:

> "oil change"

That's the entire answer. Given a top context chunk that's eleven repetitions of the word "oil" with no real sentence structure, the model's best grounded response is to just echo the query back. This is the clearest demonstration in the series of why upstream ranking quality matters beyond an abstract metric: Part 4's reranking mistake doesn't stay contained in a ranking table, it flows all the way through to a generated answer a real user would see and finds completely useless.

---

## What this means in practice

- **Don't ask a single generation to do two jobs it can't reliably hold together.** "Answer, but abstain if you shouldn't" is two decisions compressed into one instruction; splitting them into an explicit check-then-generate pipeline was the difference between a reliable abstention and a confident hallucination, on the exact same model.
- **Don't trust a model to format its own evidence trail.** Citation markers are exactly the kind of mechanical, structural output a small model will drop under instruction load. Attributing citations programmatically after generation is more reliable, auditable, and — reusing Part 1's tokenizer — required almost no new code.
- **Citation attribution and claim verification are different problems.** This part only solved attribution (which source contributed which words). It did not solve verification (is this specific claim actually true according to that source) — the SKU/oil fabrication shows exactly where that gap bites. Verification needs its own dedicated check, which is where Part 6's evaluation stage picks up.
- **Generation inherits every upstream mistake.** The "oil change" case closes a loop that's been open since Part 1: a keyword-stuffing weakness first identified in TF-IDF, then shown to also fool this part's cross-encoder reranker, ultimately surfaces as a degenerate, useless answer at the very end of the pipeline. Nothing downstream fixed it, because nothing downstream was built to check for it.

---

## What's next

Part 6 builds the evaluation layer this series has been implicitly asking for since Part 2: retrieval recall@k over the known-correct answers established in every part so far, and a groundedness check that tries to catch exactly the SKU/oil-change-style fabrication this part surfaced. Neither turns out to be a clean win — recall@k reveals that reranking actually hurt overall accuracy on this query set, and the groundedness judge has its own, different blind spots.

Full code for this post, and the rest of the series: [github.com/Thimmasani/rag-from-scratch](https://github.com/Thimmasani/rag-from-scratch) — see `retrieval/generation.py`, `pipeline.py` (the full Parts 1-5 pipeline).
