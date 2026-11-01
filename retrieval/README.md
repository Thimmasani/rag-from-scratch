# Retrieval code

Code for the "Building RAG From Scratch" series. Each part's code lives
here as plain scripts you can run directly, no notebook required.

## Setup

Part 1 (TF-IDF, BM25) is pure Python, standard library only -- no install
needed beyond Python 3.10+.

Part 2 onward uses `sentence-transformers` for embeddings, which pulls in
PyTorch. Set up a virtual environment so this doesn't pollute your system
Python:

```bash
cd retrieval
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## About the models (no large files committed)

Two models get downloaded automatically, neither is committed to the repo
-- model weights are regenerable, versioned artifacts, not source code,
and don't belong in git history:

- Part 2: `sentence-transformers/all-MiniLM-L6-v2` (~90MB), a bi-encoder
  for embeddings.
- Part 4: `cross-encoder/ms-marco-MiniLM-L-6-v2` (~130MB), a cross-encoder
  for reranking.
- Part 5: `Qwen/Qwen2.5-0.5B-Instruct` (~1GB), a small instruction-tuned
  model for generation, run locally via `transformers` (already a
  dependency of `sentence-transformers`, no extra install needed).

Each downloads and caches automatically the first time you run a script
that uses it:

```bash
source .venv/bin/activate
python3 embeddings.py   # downloads the bi-encoder on first run
python3 reranker.py     # downloads the cross-encoder on first run
python3 generation.py   # downloads the generation model on first run
```

The first run of each downloads the model to Hugging Face's local cache
(`~/.cache/huggingface` by default) and may take a few seconds to a
minute depending on your connection. Every run after that loads instantly
from the local cache, fully offline.

If you want to pre-download either model without running a full script
(e.g. to warm the cache before a demo):

```bash
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
python3 -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
```

To use a different model, change `model_name` in `EmbeddingRetriever`
(`embeddings.py`) or `CrossEncoderReranker` (`reranker.py`) -- any model on
the [sentence-transformers model hub](https://huggingface.co/models?library=sentence-transformers)
works the same way.

## Running each part

```bash
# Part 1 -- lexical retrieval (TF-IDF, BM25)
python3 tfidf.py          # TF-IDF scores on the toy corpus
python3 bm25.py            # BM25 scores on the toy corpus
python3 compare.py         # side-by-side TF-IDF vs BM25 table
python3 query.py           # interactive: type your own queries, see live scores

# Part 2 -- semantic retrieval (embeddings)
python3 embeddings.py      # embedding-based scores on the toy corpus
python3 compare_semantic.py  # three-way table + wrong-document experiment

# Part 3 -- hybrid retrieval (Reciprocal Rank Fusion)
python3 rrf.py              # BM25 + embeddings fused by rank
python3 compare_hybrid.py   # BM25 vs embeddings vs RRF, including the tie case

# Part 4 -- cross-encoder reranking
python3 reranker.py         # rerank hybrid candidates with a cross-encoder
python3 compare_reranked.py # before/after tables, including two new failure modes

# Part 5 -- generation (abstention + citation attribution)
python3 generation.py       # grounded answer, abstention, and citation examples
python3 pipeline.py         # the FULL pipeline: Parts 1-5 wired together
```

`compare_semantic.py` runs two things: the three-way TF-IDF/BM25/embeddings
comparison on the same 6-document corpus Part 1 uses (so those numbers stay
reproducible), and a separate wrong-document experiment on an *extended*
7-document corpus (`DOCUMENTS_WITH_INVOICE` in `corpus.py`) that adds an
invoice-number document to show embeddings routing a bare ID to the wrong
document. The extra document is kept out of the main corpus deliberately --
adding it there would shift every IDF value and break Part 1's hand-worked
calculations.

## Files

| File | Part | What it is |
|---|---|---|
| `corpus.py` | 1+ | Shared toy corpus + queries used across all parts |
| `tfidf.py` | 1 | TF-IDF retriever, from scratch (no sklearn) |
| `bm25.py` | 1 | BM25 retriever, from scratch (no rank_bm25) |
| `compare.py` | 1 | TF-IDF vs BM25 side-by-side table |
| `query.py` | 1 | Interactive CLI to try your own queries (lexical only) |
| `embeddings.py` | 2 | Embedding-based retriever using sentence-transformers |
| `compare_semantic.py` | 2 | TF-IDF vs BM25 vs embeddings three-way table |
| `rrf.py` | 3 | Reciprocal Rank Fusion + `HybridRetriever` combining any retrievers |
| `compare_hybrid.py` | 3 | BM25 vs embeddings vs RRF, including the exact-tie case |
| `reranker.py` | 4 | `CrossEncoderReranker`, reranks a short candidate list |
| `compare_reranked.py` | 4 | Hybrid vs reranked, including two new reranker failure modes |
| `generation.py` | 5 | `Generator`: relevance check, grounded generation, citation attribution |
| `pipeline.py` | 5 | The full Parts 1-5 pipeline, run end to end on every series query |
