"""
Toy corpus shared by TF-IDF and BM25 implementations.

Deliberately small and hand-inspectable: every score the algorithms produce
on this corpus can be verified by hand, which is the point of a from-scratch
walkthrough (you should never have to "trust" the code).

The corpus is designed to showcase specific retrieval behaviors used in the
article:
  - doc 0, 1: near-duplicate topic (cars) with different vocab -> vocabulary
    mismatch case where lexical search struggles but semantic search wins.
  - doc 2: contains a rare exact-match token (an "ID") -> case where lexical
    search wins and semantic search can fail.
  - doc 3, 4: generic distractors to make IDF matter (common words appear in
    almost every doc, rare words appear in just one).
  - doc 5: a long, keyword-stuffed document that repeats "oil" many times ->
    demonstrates TF-IDF's lack of term-frequency saturation (it over-rewards
    repetition) vs BM25, which saturates and keeps a short, naturally
    on-topic document competitive.
"""

DOCUMENTS = [
    # 0
    "the car needs an oil change and new brake pads",
    # 1
    "my automobile requires fresh engine oil and brake service",
    # 2
    "replacement part SKU-48213-B is on back order until next week",
    # 3
    "the weather today is sunny with a light breeze in the afternoon",
    # 4
    "the car dealership offers free oil change on the weekend",
    # 5
    "oil oil oil best oil change deals oil discount oil coupon oil near me "
    "oil prices oil service oil special oil offer cheap oil oil shop",
]

DOC_LABELS = [
    "doc0: car / oil / brakes (lexical phrasing A, natural, concise)",
    "doc1: automobile / oil / brakes (lexical phrasing B, same topic)",
    "doc2: exact SKU code (rare-token case)",
    "doc3: unrelated distractor (weather)",
    "doc4: car / oil, overlaps doc0 vocabulary",
    "doc5: 'oil' repeated 11x, keyword-stuffed, low real content",
]

QUERIES = [
    "car oil change",
    "automobile brake service",
    "SKU-48213-B",
    "weekend weather",
    "oil change",
]

# Extra queries for Part 2 (semantic retrieval): probing how embeddings
# handle an exact code that was NOT the one actually in the corpus, with
# and without surrounding natural-language context. See writing/02 for
# the discussion -- embeddings don't fail on exact codes outright, they
# degrade gracefully by latching onto surrounding context words instead
# of the code itself.
UNSEEN_CODE_QUERIES = [
    "SKU-48213-B",  # the real code, present in doc2
    "XJQ-99281-Z",  # a structurally similar but entirely unseen code, alone
    "part number XJQ-99281-Z is on back order",  # unseen code + context
]
