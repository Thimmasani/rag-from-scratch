"""
Vectorless RAG evaluation driver, Part 12 of the RAG-from-scratch series.

Compares tree navigation (vectorless.py) against flat chunk-embedding
retrieval (Part 2's EmbeddingRetriever, over the same document flattened
into 9 leaf chunks) on the same 6 queries -- all of which depend on
knowing WHICH PLAN'S section a near-identical sentence belongs to, not
just matching words.
"""

from embeddings import EmbeddingRetriever
from generation import Generator
from vectorless import navigate_tree
from vectorless_corpus import DOCUMENT_TREE, QUERIES, flatten_leaves


def run_comparison():
    generator = Generator()

    leaves = flatten_leaves(DOCUMENT_TREE)
    leaf_paths = [path for path, _text in leaves]
    leaf_texts = [text for _path, text in leaves]
    emb = EmbeddingRetriever(leaf_texts)

    tree_correct = 0
    flat_correct = 0
    fallback_count = 0
    n = len(QUERIES)

    for query, (expected_section, expected_fact) in QUERIES.items():
        tree_path, tree_text, used_fallback = navigate_tree(query, generator)
        fallback_count += used_fallback
        tree_ok = expected_section in tree_path and expected_fact in tree_text

        flat_idx, flat_score = emb.search(query, top_k=1)[0]
        flat_path, flat_text = leaf_paths[flat_idx], leaf_texts[flat_idx]
        flat_ok = expected_section in flat_path and expected_fact in flat_text

        tree_correct += tree_ok
        flat_correct += flat_ok

        print(f'Query: "{query}"')
        print(f"  expected: {expected_section} / contains '{expected_fact}'")
        print(f"  tree nav  -> {tree_path}  (fallback used: {used_fallback})  correct={tree_ok}")
        print(f"    text: {tree_text}")
        print(f"  flat emb  -> {flat_path}  (score {flat_score:.4f})  correct={flat_ok}")
        print(f"    text: {flat_text}")
        print()

    print(f"Tree navigation: {tree_correct}/{n} correct ({fallback_count} queries used a navigation fallback).")
    print(f"Flat embeddings: {flat_correct}/{n} correct.")


if __name__ == "__main__":
    run_comparison()
