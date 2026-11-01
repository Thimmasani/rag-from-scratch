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
│   ├── rrf.py              # Part 3: Reciprocal Rank Fusion / hybrid retrieval
│   ├── reranker.py          # Part 4: cross-encoder reranking
│   ├── generation.py         # Part 5: generation, abstention, citations
│   ├── pipeline.py            # Part 5: the full Parts 1-5 pipeline, end to end
│   ├── eval_retrieval.py        # Part 6: recall@k across every retrieval stage
│   ├── eval_generation.py        # Part 6: LLM-as-judge groundedness evaluation
│   ├── compare.py                  # Part 1: TF-IDF vs BM25 table
│   ├── compare_semantic.py          # Part 2: TF-IDF vs BM25 vs embeddings table
│   ├── compare_hybrid.py             # Part 3: BM25 vs embeddings vs RRF table
│   ├── compare_reranked.py            # Part 4: hybrid vs reranked table
│   ├── query.py                         # interactive CLI (lexical)
│   └── README.md                         # setup, including model download instructions
└── writing/           # article drafts, one per series part
    ├── 01_tfidf_bm25.md
    ├── 02_embeddings.md
    ├── 03_hybrid_rrf.md
    ├── 04_reranking.md
    ├── 05_generation.md
    ├── 06_evaluation.md
    └── rag_interview_questions.md  # companion Q&A piece, not a numbered part
```

## Series plan

1. **Lexical retrieval** — TF-IDF, BM25, why BM25 wins. ✅ published
2. **Semantic retrieval** — embeddings, where they fix lexical failures, where they don't. ✅ published
3. **Hybrid retrieval** — BM25 + semantic fusion (RRF), and RRF's own tie-breaking blind spot. ✅ published
4. **Reranking** — cross-encoders fix one RRF tie cleanly, but introduce two new failure modes of their own. ✅ published
5. **The generation layer** — abstention and citations both fail as single-shot instructions and need to be split into separate steps; citation attribution isn't claim verification. ✅ published
6. **Evaluating the whole pipeline** — recall@k shows reranking actually *hurts* overall accuracy on this query set; an LLM-as-judge groundedness check has its own distinct blind spots. ✅ published — last of the originally planned 6 parts.
7. (possible stretch) Query rewriting/expansion, NLI-based groundedness checking

**Companion piece (not a numbered part):** [RAG Interview Questions](https://dineshreddythimmasani.com/blog/rag-interview-questions/) — easy to senior level, every answer backed by a specific number or failure case from Parts 1-6 rather than a generic textbook answer.

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

python3 rrf.py                # Part 3, hybrid retrieval (BM25 + embeddings)
python3 compare_hybrid.py

python3 reranker.py           # Part 4, downloads a cross-encoder model on first run
python3 compare_reranked.py

python3 generation.py         # Part 5, downloads a small generation model on first run
python3 pipeline.py           # the full Parts 1-5 pipeline, run on every series query

python3 eval_retrieval.py     # Part 6, recall@k across every retrieval stage
python3 eval_generation.py    # Part 6, LLM-as-judge groundedness evaluation
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
