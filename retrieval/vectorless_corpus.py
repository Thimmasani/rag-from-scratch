"""
Fixture for Part 12 (vectorless RAG / tree navigation). A single
hierarchical service agreement: 3 plans (Residential, Business,
Enterprise), each with the SAME 3 subsections (Pricing, Fees,
Cancellation Terms), deliberately near-identical in wording across
plans and differing mainly in the numbers -- the structural case
vectorless/tree-navigation approaches are pitched for: the right answer
depends on which SECTION you're in, not on which sentence is most
semantically similar to the query, since the sentences are almost the
same sentence three times over.
"""

DOCUMENT_TREE = {
    "title": "Service Agreement",
    "children": [
        {
            "title": "1. Residential Plan",
            "children": [
                {"title": "1.1 Pricing", "text": "The Residential Plan costs $19.99 per month, billed monthly."},
                {"title": "1.2 Fees", "text": "The Residential Plan has an early termination fee of $50 if cancelled before the end of the first 3 months."},
                {"title": "1.3 Cancellation Terms", "text": "Residential Plan cancellations take effect at the end of the current billing month, with no partial refund for the remaining days."},
            ],
        },
        {
            "title": "2. Business Plan",
            "children": [
                {"title": "2.1 Pricing", "text": "The Business Plan costs $89.99 per month, billed monthly."},
                {"title": "2.2 Fees", "text": "The Business Plan has an early termination fee of $250 if cancelled before the end of the first 6 months."},
                {"title": "2.3 Cancellation Terms", "text": "Business Plan cancellations take effect at the end of the current billing month, with a prorated refund for unused enterprise support credits only."},
            ],
        },
        {
            "title": "3. Enterprise Plan",
            "children": [
                {"title": "3.1 Pricing", "text": "The Enterprise Plan costs $499.99 per month, billed annually in advance."},
                {"title": "3.2 Fees", "text": "The Enterprise Plan has an early termination fee of $2,000 if cancelled before the end of the first 12 months."},
                {"title": "3.3 Cancellation Terms", "text": "Enterprise Plan cancellations take effect at the end of the current billing month, with a full refund for any fully unused billing quarters remaining under the annual term."},
            ],
        },
    ],
}


def flatten_leaves(node: dict, path: str = "") -> list[tuple[str, str]]:
    """Returns [(path, leaf_text), ...] -- used both to build a flat
    chunk list for embedding retrieval, and to look up a leaf's section
    path once a tree-navigation run reaches one."""
    current_path = f"{path} > {node['title']}" if path else node["title"]
    if "text" in node:
        return [(current_path, node["text"])]
    leaves = []
    for child in node.get("children", []):
        leaves.extend(flatten_leaves(child, current_path))
    return leaves


# query -> (expected section path substring, expected fact substring)
# -- correctness means the retrieved text contains the right fact AND
# it came from the right plan's section, not a different plan's
# near-identical sentence.
QUERIES = {
    "What is the early termination fee for the Business plan?": ("2. Business Plan", "$250"),
    "What is the early termination fee for the Residential plan?": ("1. Residential Plan", "$50"),
    "What is the early termination fee for the Enterprise plan?": ("3. Enterprise Plan", "$2,000"),
    "How much does the Business plan cost per month?": ("2. Business Plan", "$89.99"),
    "What happens to unused credits if I cancel the Business plan?": ("2. Business Plan", "enterprise support credits"),
    "What refund do I get if I cancel the Enterprise plan?": ("3. Enterprise Plan", "fully unused billing quarters"),
}
