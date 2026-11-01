"""
Multi-hop retrieval evaluation, Part 9 of the RAG-from-scratch series.

Two experiments against the same three fact-chain queries:
  1. Recall for "both hop documents retrieved" -- single-shot vs. decomposed.
  2. End-to-end answer correctness -- does Generator.answer() on each
     approach's retrieved context actually produce the right composite
     answer, checked via simple substring match against the expected text.
"""

from bm25 import Bm25Retriever
from embeddings import EmbeddingRetriever
from rrf import HybridRetriever
from generation import Generator
from multihop_corpus import DOCUMENTS, DOC_LABELS, GROUND_TRUTH_MULTIHOP
from multihop import single_shot_retrieve, decompose_and_retrieve


def both_hops_retrieved(retrieved: list[int], hop1_doc: int, hop2_doc: int) -> bool:
    return hop1_doc in retrieved and hop2_doc in retrieved


def recall_comparison(hybrid: HybridRetriever, generator: Generator, top_k: int = 4):
    print("=== 1. Recall: are BOTH hop documents retrieved? ===\n")
    single_hits = 0
    decomposed_hits = 0
    n = len(GROUND_TRUTH_MULTIHOP)

    for query, info in GROUND_TRUTH_MULTIHOP.items():
        single = single_shot_retrieve(query, hybrid, top_k=top_k)
        decomposed, debug = decompose_and_retrieve(query, hybrid, generator, top_k_per_hop=top_k // 2 + 1)

        single_ok = both_hops_retrieved(single, info["hop1_doc"], info["hop2_doc"])
        decomposed_ok = both_hops_retrieved(decomposed, info["hop1_doc"], info["hop2_doc"])
        single_hits += single_ok
        decomposed_hits += decomposed_ok

        print(f'Query: "{query}"')
        print(f"  single-shot top-{top_k}:  {single}  -> both hops retrieved: {single_ok}")
        print(f"  hop1 sub-query: \"{debug['hop1_query']}\"{' (fallback to original)' if debug['hop1_fallback'] else ''}")
        print(f"  hop2 sub-query: \"{debug['hop2_query']}\"{' (fallback to original)' if debug['hop2_fallback'] else ''}")
        print(f"  decomposed union: {decomposed}  -> both hops retrieved: {decomposed_ok}")
        print()

    print(f"Single-shot: {single_hits}/{n} chains with both hop docs retrieved.")
    print(f"Decomposed:  {decomposed_hits}/{n} chains with both hop docs retrieved.\n")


def answer_correctness(hybrid: HybridRetriever, generator: Generator, top_k: int = 4):
    print("=== 2. End-to-end answer correctness ===\n")
    single_correct = 0
    decomposed_correct = 0
    n = len(GROUND_TRUTH_MULTIHOP)

    for query, info in GROUND_TRUTH_MULTIHOP.items():
        expected = info["expected_answer_text"]

        single_indices = single_shot_retrieve(query, hybrid, top_k=top_k)
        single_context = [DOCUMENTS[i] for i in single_indices]
        single_result = generator.answer(query, single_context)
        single_ok = expected.lower() in single_result["answer"].lower()
        single_correct += single_ok

        decomposed_indices, _debug = decompose_and_retrieve(query, hybrid, generator, top_k_per_hop=top_k // 2 + 1)
        decomposed_context = [DOCUMENTS[i] for i in decomposed_indices]
        decomposed_result = generator.answer(query, decomposed_context)
        decomposed_ok = expected.lower() in decomposed_result["answer"].lower()
        decomposed_correct += decomposed_ok

        print(f'Query: "{query}" (expected: contains "{expected}")')
        print(f"  single-shot answer:  {single_result['answer']!r}  -> correct: {single_ok}")
        print(f"  decomposed answer:   {decomposed_result['answer']!r}  -> correct: {decomposed_ok}")
        print()

    print(f"Single-shot: {single_correct}/{n} correct composite answers.")
    print(f"Decomposed:  {decomposed_correct}/{n} correct composite answers.")


if __name__ == "__main__":
    bm25 = Bm25Retriever(DOCUMENTS)
    emb = EmbeddingRetriever(DOCUMENTS)
    hybrid = HybridRetriever([bm25, emb])
    generator = Generator()

    recall_comparison(hybrid, generator)
    answer_correctness(hybrid, generator)
