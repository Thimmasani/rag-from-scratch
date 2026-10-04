# RAG From Scratch

A series building a full RAG pipeline (retrieval → reranking → generation →
evaluation) without framework abstractions (no LangChain `VectorStore`/
`Retriever`) doing the work for you -- paired with articles explaining the
math and design decisions behind each stage.

Live site: [dineshreddythimmasani.com](https://dineshreddythimmasani.com)

## Structure

```
.
├── src/              # Astro site (this repo's root doubles as the site)
├── public/
├── retrieval/         # Python retrieval code for each part
│   ├── corpus.py       # shared toy corpus + queries
│   ├── tfidf.py         # Part 1: TF-IDF from scratch
│   ├── bm25.py           # Part 1: BM25 from scratch
│   ├── embeddings.py      # Part 2: semantic retrieval (sentence-transformers)
│   ├── compare.py          # Part 1: TF-IDF vs BM25 table
│   ├── compare_semantic.py  # Part 2: TF-IDF vs BM25 vs embeddings table
│   ├── query.py              # interactive CLI (lexical)
│   └── README.md              # setup, including model download instructions
└── writing/           # article drafts, one per series part
    ├── 01_tfidf_bm25.md
    └── 02_embeddings.md
```

## Series plan

1. **Lexical retrieval** — TF-IDF, BM25, why BM25 wins. ✅ published
2. **Semantic retrieval** — embeddings, where they fix lexical failures, where they don't. ✅ published
3. Hybrid retrieval — BM25 + semantic fusion (RRF)
4. Reranking — cross-encoders, ColBERT
5. The generation layer — prompt construction, grounding, citations
6. Evaluating RAG — recall@k, groundedness scoring
7. (stretch) Query rewriting/expansion, production lessons learned

## Running the retrieval code

See [`retrieval/README.md`](./retrieval/README.md) for setup, including how
the embedding model is downloaded (not committed to this repo -- model
weights are regenerable, cached locally on first run).

Quick start:

```bash
cd retrieval
python3 tfidf.py       # Part 1, no install needed
python3 bm25.py
python3 compare.py

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 embeddings.py   # Part 2, downloads ~90MB model on first run
python3 compare_semantic.py
```

## Running the site locally

```bash
npm install
npm run dev       # localhost:4321
npm run build     # outputs to dist/
npm run preview   # serve the production build locally
```

## Deployment

The live site is self-hosted on an Oracle Cloud VM behind Caddy
(automatic HTTPS via Let's Encrypt), with the domain pointed at it via
GoDaddy DNS. To deploy a new build:

```bash
npm run build
rsync -avz --delete dist/ <user>@<host>:/var/www/dineshreddythimmasani/
```
