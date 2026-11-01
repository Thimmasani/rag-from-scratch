"""
Vectorless RAG via tree navigation, Part 12 of the RAG-from-scratch
series.

No embeddings, no chunking, no vector index. The document is already a
tree (section -> subsection -> leaf text). At query time, the model reads
the CHILD TITLES at the current level and picks which one to descend
into, repeating until it reaches a leaf -- navigating structure the way
a person uses a table of contents, rather than comparing embedded
fragments by similarity.
"""

from generation import Generator
from vectorless_corpus import DOCUMENT_TREE


_LETTERS = "ABCDEFGHIJ"


def _choose_child(query: str, node: dict, generator: Generator) -> tuple[dict, bool]:
    """Returns (chosen_child, used_fallback). Asks the model to pick one
    of this node's child titles; falls back to the first child if the
    model's answer doesn't clearly match any title -- a recoverable
    failure, not a crash.

    Uses LETTER labels (A/B/C), not numbers -- an earlier version used
    numbers, which collided with this fixture's own numbered section
    titles ("1. Residential Plan", "2. Business Plan", ...), producing a
    prompt that effectively read "1. 1. Residential Plan / 2. 2. Business
    Plan" and caused the model to anchor on the first digit regardless of
    the actual question, every single time (verified directly: the same
    prompt with letters instead of numbers correctly varies its answer
    with the question). That was a prompt bug, not a finding about the
    technique -- worth remembering as a lesson on its own: always check
    whether a "the model fails every time" result survives a basic
    presentation change before reporting it as a technique's limitation."""
    children = node["children"]
    titles = [c["title"] for c in children]
    lettered = "\n".join(f"{_LETTERS[i]}. {t}" for i, t in enumerate(titles))
    prompt = f"""Question: {query}

Which of these sections is most likely to contain the answer? Respond with
ONLY the letter of the section, nothing else.

Sections:
{lettered}"""
    response = generator.generate_raw(prompt, max_new_tokens=5).strip().upper()

    for i, title in enumerate(titles):
        if _LETTERS[i] in response:
            return children[i], False

    # Fallback: no clean letter match -- try substring match on title text.
    for child in children:
        if child["title"].lower() in response.lower():
            return child, False

    return children[0], True


def navigate_tree(query: str, generator: Generator, root: dict = DOCUMENT_TREE) -> tuple[str, str, bool]:
    """Descends the tree top-down. Returns (section_path, leaf_text, any_fallback_used)."""
    node = root
    path = node["title"]
    any_fallback = False

    while "children" in node:
        node, used_fallback = _choose_child(query, node, generator)
        any_fallback = any_fallback or used_fallback
        path = f"{path} > {node['title']}"

    return path, node["text"], any_fallback
