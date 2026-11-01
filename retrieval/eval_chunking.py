"""
Chunking evaluation, Part 7 of the RAG-from-scratch series.

Three experiments:
  1. Boundary-splitting demo: find a chunk_size (overlap=0) where one
     document's fact sentence gets split across two chunks, then show
     that a modest overlap keeps it intact.
  2. Chunk-size sweep: recall@k for fixed-size chunking at several chunk
     sizes, measured against GROUND_TRUTH_CHUNKING's literal fact text
     (not document index -- see chunking_corpus.py's docstring for why).
  3. Fixed-size vs sentence-aware chunking at a matched average size.
"""

from chunking import fixed_size_chunks, sentence_aware_chunks
from chunking_corpus import DOCUMENTS, DOC_LABELS, QUERIES, GROUND_TRUTH_CHUNKING
from embeddings import EmbeddingRetriever


def chunk_corpus(chunk_fn, **kwargs) -> tuple[list[str], list[int]]:
    """Apply chunk_fn to every document, returning the flat list of chunks
    plus a parallel list of which original document each chunk came from."""
    all_chunks = []
    source_doc = []
    for doc_idx, doc in enumerate(DOCUMENTS):
        for chunk in chunk_fn(doc, **kwargs):
            all_chunks.append(chunk)
            source_doc.append(doc_idx)
    return all_chunks, source_doc


def recall_at_k_by_fact(retriever: EmbeddingRetriever, chunks: list[str], query: str, fact_text: str, k: int) -> bool:
    results = retriever.search(query, top_k=k)
    return any(fact_text in chunks[idx] for idx, _score in results)


def boundary_splitting_demo():
    print("=== 1. Boundary-splitting demo ===\n")
    doc_idx = 1  # returns/shipping doc, fact in the middle
    doc = DOCUMENTS[doc_idx]
    fact = GROUND_TRUTH_CHUNKING["When must a return be initiated to get a full refund?"]["fact_text"]
    fact_len = len(fact.split())
    print(f'Fact ({fact_len} words): "{fact}"\n')

    # A chunk_size smaller than the fact can NEVER contain it regardless of
    # overlap -- search from fact_len upward for the smallest chunk_size
    # where a bad window alignment still splits it at overlap=0.
    split_size = None
    for chunk_size in range(fact_len, fact_len + 15):
        chunks = fixed_size_chunks(doc, chunk_size, overlap=0)
        if not any(fact in chunk for chunk in chunks):
            split_size = chunk_size
            break

    if split_size is None:
        print("No chunk_size in the tested range split the fact -- skipping demo.")
        return

    print(f"chunk_size={split_size} (just {split_size - fact_len} words more than the fact itself), overlap=0:")
    chunks = fixed_size_chunks(doc, split_size, overlap=0)
    for i, chunk in enumerate(chunks):
        has_fact_words = any(w in chunk for w in ("45", "refund."))
        marker = "  <-- part of the fact lands here, but not the whole thing" if has_fact_words else ""
        print(f"  [{i}] {chunk}{marker}")
    print(f"  Fact intact in any single chunk? {any(fact in c for c in chunks)}")

    # Find the minimal overlap (same chunk_size) that restores the fact intact.
    fixed_overlap = None
    for overlap in range(1, split_size):
        chunks_overlap = fixed_size_chunks(doc, split_size, overlap=overlap)
        if any(fact in chunk for chunk in chunks_overlap):
            fixed_overlap = overlap
            break

    if fixed_overlap is None:
        print("\nNo overlap in range restored the fact -- unexpected, worth a second look.")
        return

    print(f"\nSame chunk_size={split_size}, overlap={fixed_overlap} (minimal overlap that fixes it):")
    chunks_overlap = fixed_size_chunks(doc, split_size, overlap=fixed_overlap)
    for i, chunk in enumerate(chunks_overlap):
        marker = "  <-- fact intact here" if fact in chunk else ""
        print(f"  [{i}] {chunk}{marker}")


def chunk_size_sweep():
    print("\n=== 2. Chunk-size sweep: recall@1 vs recall@3, fixed-size chunking ===\n")
    for chunk_size in [15, 30, 60, 120]:
        chunks, _ = chunk_corpus(fixed_size_chunks, chunk_size=chunk_size, overlap=0)
        retriever = EmbeddingRetriever(chunks)
        hits_1 = hits_3 = 0
        for query, info in GROUND_TRUTH_CHUNKING.items():
            if recall_at_k_by_fact(retriever, chunks, query, info["fact_text"], 1):
                hits_1 += 1
            if recall_at_k_by_fact(retriever, chunks, query, info["fact_text"], 3):
                hits_3 += 1
        n = len(GROUND_TRUTH_CHUNKING)
        print(
            f"  chunk_size={chunk_size:>4} words  ({len(chunks)} chunks total)  "
            f"recall@1: {hits_1 / n:.2f} ({hits_1}/{n})  recall@3: {hits_3 / n:.2f} ({hits_3}/{n})"
        )


def fixed_vs_sentence_aware():
    print("\n=== 3. Fixed-size vs sentence-aware chunking, matched average size ===\n")
    target_size = 40
    sa_chunks, _ = chunk_corpus(sentence_aware_chunks, target_size=target_size)
    avg_sa_len = sum(len(c.split()) for c in sa_chunks) / len(sa_chunks)
    print(f"Sentence-aware (target_size={target_size}): {len(sa_chunks)} chunks, avg length {avg_sa_len:.1f} words")

    fixed_size = round(avg_sa_len)
    fixed_chunks, _ = chunk_corpus(fixed_size_chunks, chunk_size=fixed_size, overlap=0)
    print(f"Fixed-size (chunk_size={fixed_size}, matched to that average): {len(fixed_chunks)} chunks\n")

    retrievers = {
        "sentence-aware": (sa_chunks, EmbeddingRetriever(sa_chunks)),
        "fixed-size": (fixed_chunks, EmbeddingRetriever(fixed_chunks)),
    }

    for name, (chunks, retriever) in retrievers.items():
        hits_1 = hits_3 = 0
        for query, info in GROUND_TRUTH_CHUNKING.items():
            if recall_at_k_by_fact(retriever, chunks, query, info["fact_text"], 1):
                hits_1 += 1
            if recall_at_k_by_fact(retriever, chunks, query, info["fact_text"], 3):
                hits_3 += 1
        n = len(GROUND_TRUTH_CHUNKING)
        print(f"  {name:<16} recall@1: {hits_1 / n:.2f} ({hits_1}/{n})  recall@3: {hits_3 / n:.2f} ({hits_3}/{n})")


if __name__ == "__main__":
    boundary_splitting_demo()
    chunk_size_sweep()
    fixed_vs_sentence_aware()
