"""
Chunking strategies, Part 7 of the RAG-from-scratch series.

Two strategies, both word-count-based (not token-based -- this series has
no tokenizer dependency until Part 5's generation model, and chunk text
needs to stay human-readable for the article, so splitting on whitespace-
delimited words is the from-scratch-appropriate choice here; production
systems more commonly chunk by LLM token count, which is a straightforward
substitution of the unit, not a different mechanism):

- fixed_size_chunks: a naive sliding window over words, with configurable
  overlap. Doesn't know or care about sentence boundaries -- a fact can
  land split across two chunks if the window happens to cut through it.
- sentence_aware_chunks: splits into sentences first (same regex
  generation.py already uses for citation attribution), then greedily
  packs whole sentences into chunks up to a target size. A sentence never
  gets split across a chunk boundary, at the cost of variable chunk sizes.
"""

import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def fixed_size_chunks(text: str, chunk_size: int, overlap: int = 0) -> list[str]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if not (0 <= overlap < chunk_size):
        raise ValueError(
            f"overlap must satisfy 0 <= overlap < chunk_size, got overlap={overlap}, chunk_size={chunk_size}"
        )

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


def sentence_aware_chunks(text: str, target_size: int) -> list[str]:
    if target_size <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}")

    sentences = [s for s in _SENTENCE_SPLIT.split(text.strip()) if s]
    chunks = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence.split())
        if current and current_len + sentence_len > target_size:
            chunks.append(" ".join(current))
            current = []
            current_len = 0
        current.append(sentence)
        current_len += sentence_len

    if current:
        chunks.append(" ".join(current))
    return chunks


if __name__ == "__main__":
    from chunking_corpus import DOCUMENTS

    sample = DOCUMENTS[1]
    print("Fixed-size (chunk_size=20, overlap=0):")
    for i, chunk in enumerate(fixed_size_chunks(sample, 20, 0)):
        print(f"  [{i}] ({len(chunk.split())} words) {chunk}")

    print("\nSentence-aware (target_size=40):")
    for i, chunk in enumerate(sentence_aware_chunks(sample, 40)):
        print(f"  [{i}] ({len(chunk.split())} words) {chunk}")
