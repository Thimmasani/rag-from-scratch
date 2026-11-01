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

Every part's main comparisons -- Parts 1, 2, 3, 4, and 6's recall@k eval --
run against this exact 6-document DOCUMENTS list, so every score printed
across the whole series stays reproducible. Parts 2's wrong-document
experiment uses a SEPARATE, extended corpus (DOCUMENTS_WITH_INVOICE below)
rather than appending to DOCUMENTS directly -- appending here would change
every IDF value in Part 1's hand-worked calculations, since IDF depends on
total document count.
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

# --- Part 2 wrong-document experiment: a separate, extended corpus -------
#
# doc6 (invoice number) is a second code-bearing document in a totally
# different domain from doc2's SKU code, but sharing the same numeric
# digits -> used to show embeddings routing a bare ID to the wrong document
# when the only signal distinguishing them is a prefix convention ("SKU-"
# vs "INV-") the model doesn't weight correctly. Kept separate from
# DOCUMENTS above so Part 1's lexical scores stay exactly reproducible.
DOCUMENTS_WITH_INVOICE = DOCUMENTS + [
    "invoice number INV-77213-K was paid on March 3rd",
]

DOC_LABELS_WITH_INVOICE = DOC_LABELS + [
    "doc6: invoice number (billing, different domain than doc2's SKU)",
]

# Bare invoice-style ID that shares digits with doc2's SKU code, used
# against DOCUMENTS_WITH_INVOICE to show embeddings routing to the WRONG
# document. See writing/02_embeddings.md for the discussion.
WRONG_DOC_QUERIES = [
    "INV-48213-B",            # bare invoice-style ID, digits match doc2's SKU
    "invoice INV-48213-B",    # same ID, with one word of context added
    "INV-48213-B was paid",   # same ID, with more natural-language context
]

# Part 3 (hybrid retrieval / RRF): a query mixing a semantic signal
# ("automobile", "service") with an exact code string. Embeddings alone
# favor doc2 (the SKU doc) because the exact code dominates the embedding;
# BM25 alone strongly favors doc1 (the actual intended document, matching
# "automobile" and "service"). Fusing them pulls doc2's wrongful lead back
# to an exact tie with doc1 -- not a clean win, which is itself the point:
# RRF ties whenever two retrievers swap which of two documents ranks #1 vs
# #2, regardless of how different their underlying scores actually are.
# See writing/03_hybrid_rrf.md.
HYBRID_RESCUE_QUERY = "automobile SKU-48213-B service"

# --- Part 6 (evaluation): ground truth labels -----------------------------
#
# The single "correct" document index (into DOCUMENTS) for every query
# used across the series, established by the discussion in each part's
# article. Used for recall@k scoring in Part 6. A query mapped to None
# has NO correct document anywhere in the corpus -- the only correct
# retrieval/generation behavior is to abstain.
GROUND_TRUTH = {
    "car oil change": 0,              # Part 1: natural phrasing is the "home" doc for this query
    "automobile brake service": 1,     # Parts 1-4: consistently doc1
    "SKU-48213-B": 2,                   # Part 1: the exact code doc
    "weekend weather": 3,                # Part 1-4: the actual weather doc (lexical's worst failure)
    "oil change": 0,                      # Part 1/4: doc0 is correct; doc5 (spam) and doc4 are NOT
    HYBRID_RESCUE_QUERY: 1,                 # Part 3/4: doc1, despite doc2's exact-code distraction
    "What is the capital of France?": None,  # no doc in this corpus answers this -- must abstain
}

# Ground truth on the extended (with-invoice) corpus -- doc6 is correct
# for all three WRONG_DOC_QUERIES variants (Part 2/3).
GROUND_TRUTH_WITH_INVOICE = {
    "INV-48213-B": 6,
    "invoice INV-48213-B": 6,
    "INV-48213-B was paid": 6,
}
