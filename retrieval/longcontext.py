"""
RAG vs. long-context stuffing, Part 10 of the RAG-from-scratch series.

Reuses existing fixtures rather than inventing a third corpus, mirroring
the series' own precedent of later parts reusing earlier parts' modules
(Part 5 reused Part 1's tokenizer; this reuses Part 7's facts and Part
8's synthetic-document generator for filler): chunking_corpus.py's FACTS
supply the "needle" (the one fact that actually answers the question),
ann_corpus.py's generate_synthetic_corpus supplies filler/distractor text
to pad a context out to a controlled token length.

Context sizes here are TARGET token counts, verified against the actual
Qwen2.5-0.5B-Instruct tokenizer (not word counts, unlike Part 7's
chunking -- this part is specifically about LLM context windows, so
tokens are the correct, honest unit here).
"""

from transformers import AutoTokenizer

from ann_corpus import generate_synthetic_corpus


def build_context(needle_fact: str, position: str, target_tokens: int, tokenizer: AutoTokenizer) -> str:
    """Assembles a context string of approximately target_tokens tokens
    by repeating synthetic filler documents, inserting needle_fact at
    "start", "middle", or "end". Validates the assembled context's
    actual token count via the real tokenizer before returning it --
    the sliding-window word-count math from Part 7 doesn't apply here,
    since LLM tokenization doesn't map 1:1 to words."""
    if position not in ("start", "middle", "end"):
        raise ValueError(f'position must be "start", "middle", or "end", got {position!r}')

    filler_docs, _meta = generate_synthetic_corpus(500, seed=7)
    filler_text = " ".join(filler_docs)

    needle_tokens = len(tokenizer.encode(needle_fact))
    filler_budget = max(target_tokens - needle_tokens, 0)

    # Repeat filler text until it has enough tokens for the requested budget.
    filler_ids = tokenizer.encode(filler_text)
    while len(filler_ids) < filler_budget:
        filler_ids = filler_ids + tokenizer.encode(filler_text)
    filler_ids = filler_ids[:filler_budget]

    half = len(filler_ids) // 2
    if position == "start":
        combined_ids = tokenizer.encode(needle_fact) + filler_ids
    elif position == "end":
        combined_ids = filler_ids + tokenizer.encode(needle_fact)
    else:  # middle
        combined_ids = filler_ids[:half] + tokenizer.encode(needle_fact) + filler_ids[half:]

    return tokenizer.decode(combined_ids)
