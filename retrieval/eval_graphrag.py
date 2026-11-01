"""
GraphRAG evaluation driver, Part 11 of the RAG-from-scratch series.

Three experiments:
  1. Entity extraction accuracy: how often does the local LLM correctly
     extract team/system/cause from each incident report, checked
     against graphrag_corpus.py's known ground truth.
  2. Local search: direct graph traversal vs. flat embedding retrieval,
     on ordinary single-document lookups.
  3. Global search: community-summary reasoning vs. flat embedding
     retrieval, on the one question with no single-document answer.
"""

from embeddings import EmbeddingRetriever
from generation import Generator
from graphrag_corpus import (
    DOCUMENTS,
    ENTITIES,
    GLOBAL_ANSWER_CAUSE,
    GLOBAL_ANSWER_COUNT,
    GLOBAL_QUERY,
    LOCAL_QUERIES,
)
from graphrag import (
    build_communities,
    build_graph,
    extract_entities,
    global_search,
    local_search,
    summarize_community,
)


def extraction_accuracy(generator: Generator) -> list[dict | None]:
    print("=== 1. Entity extraction accuracy (LLM vs. known ground truth) ===\n")
    extracted = []
    correct = 0
    for i, doc in enumerate(DOCUMENTS):
        result = extract_entities(doc, generator)
        extracted.append(result)
        expected = ENTITIES[i]
        is_correct = result is not None and all(
            result[k].lower() in expected[k].lower() or expected[k].lower() in result[k].lower()
            for k in ("team", "system", "cause")
        )
        correct += is_correct
        print(f"  doc{i}: extracted={result}  expected={expected}  correct={is_correct}")

    print(f"\n  {correct}/{len(DOCUMENTS)} documents extracted correctly.\n")
    return extracted


def local_search_comparison(graph: dict, emb: EmbeddingRetriever):
    print("=== 2. Local search: graph traversal vs. flat embedding retrieval ===\n")
    for query, correct_doc in LOCAL_QUERIES.items():
        # crude entity pull from the query itself, for the graph lookup --
        # this mirrors how a real GraphRAG query pipeline extracts query
        # entities before traversing (here, string-matched for simplicity
        # since the query vocabulary is controlled)
        team = next((e["team"] for e in ENTITIES if e["team"] in query), None)
        system = next((e["system"] for e in ENTITIES if e["system"] in query), None)

        graph_result = local_search(team, system, graph) if team and system else None
        emb_result = emb.search(query, top_k=1)[0][0]

        print(f'  "{query}"')
        print(f"    graph traversal -> doc{graph_result} ({'correct' if graph_result == correct_doc else 'WRONG'})")
        print(f"    flat embeddings -> doc{emb_result} ({'correct' if emb_result == correct_doc else 'WRONG'})")
    print()


def global_search_comparison(entities: list[dict], emb: EmbeddingRetriever, generator: Generator):
    print("=== 3. Global search: community summaries vs. flat embedding retrieval ===\n")
    print(f'Question: "{GLOBAL_QUERY}"')
    print(f"Correct answer (by construction): '{GLOBAL_ANSWER_CAUSE}', {GLOBAL_ANSWER_COUNT} incidents.\n")

    communities = build_communities(entities)
    summaries = [summarize_community(cause, docs, entities) for cause, docs in communities.items()]
    print("Pre-written community summaries (indexing time):")
    for s in summaries:
        print(f"  - {s}")

    graphrag_answer = global_search(summaries, GLOBAL_QUERY, generator)
    print(f"\nGraphRAG (global search) answer: {graphrag_answer!r}")
    graphrag_correct = (
        GLOBAL_ANSWER_CAUSE.lower() in graphrag_answer.lower() and str(GLOBAL_ANSWER_COUNT) in graphrag_answer
    )
    print(f"Contains correct cause AND correct count? {graphrag_correct}")

    print("\n--- What flat embedding retrieval returns for the same question ---")
    top_results = emb.search(GLOBAL_QUERY, top_k=3)
    for idx, score in top_results:
        print(f"  {score:.4f}  doc{idx}: {DOCUMENTS[idx]}")
    print(
        "\nEach result is ONE incident. None of them, individually or as a top-3 list, states "
        "that this cause recurs 4 times across 3 teams -- that count only exists as a property "
        "of the whole corpus, which no single retrieved chunk can carry."
    )


if __name__ == "__main__":
    generator = Generator()
    emb = EmbeddingRetriever(DOCUMENTS)

    extracted = extraction_accuracy(generator)
    # Use known ground truth for graph construction regardless of
    # extraction accuracy above -- the graph/community demo is about
    # local vs. global search, not re-testing extraction a second time.
    graph = build_graph(ENTITIES)

    local_search_comparison(graph, emb)
    global_search_comparison(ENTITIES, emb, generator)
