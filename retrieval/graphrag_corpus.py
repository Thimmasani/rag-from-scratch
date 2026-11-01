"""
Fixture for Part 11 (GraphRAG). Nine short incident-report documents
across three teams, four systems, and five root causes -- designed so a
THEMATIC question ("what's the most common root cause across every
team?") has no single-document answer. Each document mentions exactly
one team, one system, and one root cause, so entity extraction has an
unambiguous ground truth to be checked against.

Deliberately NOT reusing Part 9's multihop_corpus.py: that fixture's
chains are two-hop lookups with one correct answer each -- a good test
of LOCAL graph traversal, but it has no repeated entity across many
documents, so it can't demonstrate GLOBAL search (synthesizing a pattern
that only exists in aggregate, across documents, with no single document
containing it).
"""

DOCUMENTS = [
    # 0
    "Team Atlas's checkout service had an outage under high load. Root cause: "
    "a database connection pool timeout. Resolution: the pool size was increased.",
    # 1
    "Team Orion's payment service had an outage during a traffic spike. Root "
    "cause: a database connection pool timeout. Resolution: read replicas were added.",
    # 2
    "Team Atlas's search service slowed down for all users. Root cause: a "
    "missing database index on the products table. Resolution: the index was added.",
    # 3
    "Team Nimbus's notification service failed when a downstream service "
    "stalled. Root cause: a database connection pool timeout. Resolution: a "
    "circuit breaker was added.",
    # 4
    "Team Orion's checkout service had an outage affecting new orders. Root "
    "cause: a third-party payment gateway timeout. Resolution: a fallback "
    "gateway was configured.",
    # 5
    "Team Nimbus's search service slowed down during peak hours. Root cause: a "
    "missing database index on the orders table. Resolution: a composite index was added.",
    # 6
    "Team Atlas's payment service failed under a retry storm. Root cause: a "
    "misconfigured retry policy causing a thundering herd. Resolution: "
    "exponential backoff was added.",
    # 7
    "Team Orion's notification service failed during a traffic spike. Root "
    "cause: a message queue backlog. Resolution: consumers were autoscaled.",
    # 8
    "Team Nimbus's checkout service had an outage during a flash sale. Root "
    "cause: a database connection pool timeout. Resolution: the pool size was "
    "increased and a circuit breaker was added.",
]

# Known ground truth for every document -- used to check LLM-based entity
# extraction accuracy, and as the source of truth for graph construction
# if extraction isn't reliable enough to build the graph from directly.
ENTITIES = [
    {"team": "Atlas", "system": "checkout", "cause": "database connection pool timeout"},
    {"team": "Orion", "system": "payment", "cause": "database connection pool timeout"},
    {"team": "Atlas", "system": "search", "cause": "missing database index"},
    {"team": "Nimbus", "system": "notification", "cause": "database connection pool timeout"},
    {"team": "Orion", "system": "checkout", "cause": "payment gateway timeout"},
    {"team": "Nimbus", "system": "search", "cause": "missing database index"},
    {"team": "Atlas", "system": "payment", "cause": "misconfigured retry policy"},
    {"team": "Orion", "system": "notification", "cause": "message queue backlog"},
    {"team": "Nimbus", "system": "checkout", "cause": "database connection pool timeout"},
]

# A question with no single-document answer -- requires counting across
# every document. The correct answer (by construction): "database
# connection pool timeout" accounts for 4 of 9 incidents (doc0, 1, 3, 8),
# spanning all three teams -- the dominant cross-team pattern.
GLOBAL_QUERY = "What is the most common root cause of incidents across all teams, and how many incidents does it account for?"
GLOBAL_ANSWER_CAUSE = "database connection pool timeout"
GLOBAL_ANSWER_COUNT = 4

# Ordinary single-document lookups -- each has exactly one correct source.
LOCAL_QUERIES = {
    "What caused Team Atlas's checkout service outage?": 0,
    "What caused Team Nimbus's search service slowdown?": 5,
    "What caused Team Orion's notification service failure?": 7,
}
