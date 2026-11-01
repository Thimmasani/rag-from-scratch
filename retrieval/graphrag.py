"""
GraphRAG, Part 11 of the RAG-from-scratch series.

Real GraphRAG (Microsoft Research) does three things chunk-and-embed
retrieval doesn't: (1) extract entities and relationships from documents
into a knowledge graph, (2) cluster the graph into communities via a
graph clustering algorithm (Leiden, in the original), and (3) pre-write a
summary for each community -- all at INDEXING time, before any query
arrives. At query time it picks LOCAL search (look up an entity, expand
to its neighbors) for narrow questions, or GLOBAL search (reason over
pre-written community summaries) for thematic questions that no single
document answers.

This module hand-rolls (1) via the local Qwen model (structured-output
prompting, same pattern as Part 9's query decomposition) and a plain
dict-based graph -- no new library, this is small enough to do by hand.
(2) is SIMPLIFIED: real GraphRAG runs Leiden/Louvain clustering on the
entity graph, a nontrivial algorithm in its own right (modularity
optimization over graph partitions). Reimplementing Leiden from scratch
is out of scope for a fixture this size, the same category of judgment
call as Part 8's hnswlib exception -- except here, because the graph is
small and every entity's role is known, communities can be built directly
by grouping documents that share a root-cause entity, which is what
Leiden would converge toward on a graph this small and this cleanly
clustered anyway. This simplification is stated explicitly, not hidden.
(3) uses the same local Qwen model for community summaries and for the
final global-search answer.
"""

import re
from collections import defaultdict

from generation import Generator

_EXTRACTION_PATTERN = re.compile(
    r"TEAM:\s*(?P<team>.+?)\s*\n\s*SYSTEM:\s*(?P<system>.+?)\s*\n\s*CAUSE:\s*(?P<cause>.+?)\s*(?:\n|$)",
    re.IGNORECASE,
)


def extract_entities(document: str, generator: Generator) -> dict | None:
    """LLM-based entity extraction, step (1). Returns None if the model's
    output doesn't match the requested structured format -- a recoverable
    failure, tracked and reported rather than crashing or silently
    guessing."""
    prompt = f"""Extract the team name, system name, and root cause from this incident report.
Respond in EXACTLY this format, nothing else:
TEAM: <team name>
SYSTEM: <system name>
CAUSE: <root cause, a few words>

Incident report: {document}"""
    response = generator.generate_raw(prompt, max_new_tokens=40)
    match = _EXTRACTION_PATTERN.search(response)
    if not match:
        return None
    return {
        "team": match.group("team").strip().rstrip("."),
        "system": match.group("system").strip().rstrip("."),
        "cause": match.group("cause").strip().rstrip("."),
    }


def build_graph(entities: list[dict]) -> dict[str, set[str]]:
    """Plain dict-of-sets adjacency graph. Nodes: 'doc:{i}', 'team:{name}',
    'system:{name}', 'cause:{name}'. Each document connects to its own
    team/system/cause nodes -- a document is the hub, its entities are
    the spokes."""
    graph: dict[str, set[str]] = defaultdict(set)
    for i, entity in enumerate(entities):
        doc_node = f"doc:{i}"
        for kind in ("team", "system", "cause"):
            entity_node = f"{kind}:{entity[kind]}"
            graph[doc_node].add(entity_node)
            graph[entity_node].add(doc_node)
    return graph


def local_search(team: str, system: str, graph: dict[str, set[str]]) -> int | None:
    """Narrow lookup: find the document connected to BOTH this team and
    this system node -- a direct graph traversal, no similarity scoring
    involved."""
    team_docs = {n for n in graph.get(f"team:{team}", set()) if n.startswith("doc:")}
    system_docs = {n for n in graph.get(f"system:{system}", set()) if n.startswith("doc:")}
    matches = team_docs & system_docs
    if not matches:
        return None
    return int(next(iter(matches)).split(":")[1])


def build_communities(entities: list[dict]) -> dict[str, list[int]]:
    """SIMPLIFIED community detection -- see module docstring. Groups
    document indices by shared root-cause entity, a stand-in for what
    Leiden/Louvain clustering would converge toward on a graph this small."""
    communities: dict[str, list[int]] = defaultdict(list)
    for i, entity in enumerate(entities):
        communities[entity["cause"]].append(i)
    return communities


def summarize_community(cause: str, doc_indices: list[int], entities: list[dict]) -> str:
    """Step (3), pre-written at indexing time: one line per community,
    not per document -- this is the thing global search reasons over
    instead of raw text."""
    teams = sorted({entities[i]["team"] for i in doc_indices})
    return f"Root cause '{cause}': {len(doc_indices)} incident(s), affecting teams {', '.join(teams)}."


def global_search(community_summaries: list[str], query: str, generator: Generator) -> str:
    """Reasons over pre-written community summaries, not raw documents --
    the thing a flat chunk-embedding retriever structurally cannot do,
    since no single chunk contains the cross-document count."""
    summaries_block = "\n".join(f"- {s}" for s in community_summaries)
    prompt = f"""Community summaries (each covers multiple incidents sharing one root cause):
{summaries_block}

Question: {query}

Answer using ONLY the summaries above."""
    return generator.generate_raw(prompt, max_new_tokens=60)
