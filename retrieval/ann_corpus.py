"""
Synthetic corpus generator for Part 8 (vector search at scale). Separate
from corpus.py on purpose, same reasoning as every other new fixture in
this series: this needs thousands of documents to make brute-force vs.
ANN and pre- vs. post-filtering measurable at all, and corpus.py's 6-doc
DOCUMENTS must never be touched or grown in place.

Deterministic (seeded) template generation instead of a committed data
file -- reproducible without adding a large static asset to the repo.
"""

import random

CATEGORIES = ["electronics", "home", "sports", "books", "toys"]

_PRODUCTS = {
    "electronics": ["wireless headphones", "bluetooth speaker", "laptop stand", "usb-c charger", "smart watch", "webcam", "mechanical keyboard", "power bank"],
    "home": ["ceramic mug", "throw blanket", "desk lamp", "cutting board", "storage bin", "candle set", "picture frame", "door mat"],
    "sports": ["yoga mat", "resistance bands", "water bottle", "running shoes", "jump rope", "foam roller", "gym bag", "tennis racket"],
    "books": ["mystery novel", "cookbook", "biography", "poetry collection", "science textbook", "travel guide", "graphic novel", "self-help book"],
    "toys": ["building blocks", "puzzle set", "remote control car", "stuffed animal", "board game", "action figure", "art kit", "toy train"],
}

_ADJECTIVES = ["durable", "lightweight", "compact", "premium", "affordable", "eco-friendly", "portable", "versatile"]
_FEATURES = [
    "ships with a one-year warranty",
    "is available in several colors",
    "comes with a reusable storage case",
    "ships within two business days",
    "is backed by a satisfaction guarantee",
    "includes a quick-start guide",
    "is made from recycled materials",
    "fits most standard setups",
]


def generate_synthetic_corpus(n: int, seed: int = 42) -> tuple[list[str], list[dict]]:
    """Returns (documents, metadata) where metadata[i] = {"category": str, "date": "YYYY-MM"}.

    One document in every 50 (by construction, deterministic under the
    given seed) is tagged category="clearance" with a fixed recent date,
    giving pre/post-filtering experiments a deliberately selective filter
    to work with -- roughly 2% of the corpus, not an even fifth."""
    rng = random.Random(seed)
    documents = []
    metadata = []

    for i in range(n):
        if i % 50 == 0:
            category = "clearance"
            date = "2026-09"
        else:
            category = rng.choice(CATEGORIES)
            date = f"2026-{rng.randint(1, 8):02d}"

        base_category = category if category != "clearance" else rng.choice(CATEGORIES)
        product = rng.choice(_PRODUCTS[base_category])
        adjective = rng.choice(_ADJECTIVES)
        feature = rng.choice(_FEATURES)
        doc = f"This {adjective} {product} {feature}. Item #{i} in the {base_category} category."
        if category == "clearance":
            doc += " Currently marked down as part of a clearance sale."

        documents.append(doc)
        metadata.append({"category": category, "date": date})

    return documents, metadata


if __name__ == "__main__":
    docs, meta = generate_synthetic_corpus(20, seed=42)
    for d, m in zip(docs[:10], meta[:10]):
        print(m, "->", d)
    n_clearance = sum(1 for m in meta if m["category"] == "clearance")
    print(f"\n{n_clearance}/20 tagged clearance")
