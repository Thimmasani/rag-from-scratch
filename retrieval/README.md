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

## About the embedding model (no large files committed)

Part 2 uses `sentence-transformers/all-MiniLM-L6-v2` (~90MB). This is
**not** committed to the repo -- model weights are regenerable, versioned
artifacts, not source code, and don't belong in git history.

Instead, `sentence-transformers` downloads and caches the model
automatically the first time you run any script that imports it:

```bash
source .venv/bin/activate
python3 embeddings.py
```

The first run will download the model to Hugging Face's local cache
(`~/.cache/huggingface` by default) and may take a few seconds to a
minute depending on your connection. Every run after that loads instantly
from the local cache, fully offline.

If you want to pre-download the model without running a full script
(e.g. to warm the cache before a demo):

```bash
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

To use a different embedding model, change `model_name` in
`EmbeddingRetriever.__init__` (`embeddings.py`) -- any model on the
[sentence-transformers model hub](https://huggingface.co/models?library=sentence-transformers)
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
python3 compare_semantic.py  # three-way table: TF-IDF vs BM25 vs embeddings
```

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
