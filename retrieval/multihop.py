"""
Multi-hop retrieval, Part 9 of the RAG-from-scratch series.

single_shot_retrieve treats a multi-hop question as one query against
Part 3's HybridRetriever, unmodified -- the naive baseline every other
part of this series implicitly assumed was enough.

decompose_and_retrieve asks the generation model (Part 5's Generator,
via its public generate_raw() method) to produce a first-hop sub-query,
retrieves with it, then asks for a second-hop sub-query given what the
first hop returned, and retrieves with that. Returns the union of both
hops' top candidates as the final context pool.
"""

from generation import Generator
from rrf import HybridRetriever

_LOOKS_LIKE_QUESTION_MAX_WORDS = 25


def _looks_like_a_question(text: str) -> bool:
    """Minimal sanity check on a generated sub-query: recoverable failure
    handling for decompose_and_retrieve, see module docstring and the
    article for how often this fallback actually triggers."""
    text = text.strip()
    return bool(text) and text.endswith("?") and len(text.split()) <= _LOOKS_LIKE_QUESTION_MAX_WORDS


def single_shot_retrieve(query: str, hybrid: HybridRetriever, top_k: int) -> list[int]:
    return [idx for idx, _score in hybrid.search(query, top_k=top_k)]


def decompose_and_retrieve(
    query: str,
    hybrid: HybridRetriever,
    generator: Generator,
    top_k_per_hop: int,
) -> tuple[list[int], dict]:
    """Returns (retrieved_doc_indices, debug_info). debug_info records the
    generated sub-queries and whether either hop fell back to the
    original question, so eval_multihop.py can report how often
    decomposition actually worked vs. degraded gracefully."""
    debug = {"hop1_query": None, "hop2_query": None, "hop1_fallback": False, "hop2_fallback": False}

    hop1_prompt = f"""Question: {query}

This question needs two separate facts combined. What is the FIRST, simpler
question you would need to answer before you could answer the full question?
Respond with ONLY that first question, nothing else."""
    hop1_query = generator.generate_raw(hop1_prompt, max_new_tokens=30).strip()
    if not _looks_like_a_question(hop1_query):
        hop1_query = query
        debug["hop1_fallback"] = True
    debug["hop1_query"] = hop1_query

    hop1_results = hybrid.search(hop1_query, top_k=top_k_per_hop)
    hop1_indices = [idx for idx, _score in hop1_results]

    hop2_prompt = f"""Original question: {query}

First question: {hop1_query}

Given that you now need a second fact to fully answer the original question,
what is that SECOND question? Respond with ONLY that second question, nothing else."""
    hop2_query = generator.generate_raw(hop2_prompt, max_new_tokens=30).strip()
    if not _looks_like_a_question(hop2_query):
        hop2_query = query
        debug["hop2_fallback"] = True
    debug["hop2_query"] = hop2_query

    hop2_results = hybrid.search(hop2_query, top_k=top_k_per_hop)
    hop2_indices = [idx for idx, _score in hop2_results]

    combined = list(dict.fromkeys(hop1_indices + hop2_indices))  # union, order-preserving
    return combined, debug
