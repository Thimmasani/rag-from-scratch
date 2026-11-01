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
│   ├── freshness.py                      # Part 1 addition: index freshness
│   ├── prompt_injection.py                # Part 5 addition: prompt injection test
│   ├── chunking_corpus.py, chunking.py, eval_chunking.py       # Part 7: chunking
│   ├── ann_corpus.py, ann_search.py, eval_ann.py                # Part 8: vector search at scale
│   ├── multihop_corpus.py, multihop.py, eval_multihop.py         # Part 9: multi-hop retrieval
│   ├── longcontext.py, eval_longcontext.py                        # Part 10: RAG vs long-context
│   └── README.md                         # setup, including model download instructions
└── writing/           # article drafts, one per series part
    ├── 01_tfidf_bm25.md
    ├── 02_embeddings.md
    ├── 03_hybrid_rrf.md
    ├── 04_reranking.md
    ├── 05_generation.md
    ├── 06_evaluation.md
    ├── 07_chunking.md
    ├── 08_vector_search_at_scale.md
    ├── 09_multihop_retrieval.md
    ├── 10_long_context.md
    └── rag_interview_questions.md  # companion Q&A piece, not a numbered part
```

## Series plan

1. **Lexical retrieval** — TF-IDF, BM25, why BM25 wins. ✅ published
2. **Semantic retrieval** — embeddings, where they fix lexical failures, where they don't. ✅ published
3. **Hybrid retrieval** — BM25 + semantic fusion (RRF), and RRF's own tie-breaking blind spot. ✅ published
4. **Reranking** — cross-encoders fix one RRF tie cleanly, but introduce two new failure modes of their own. ✅ published
5. **The generation layer** — abstention and citations both fail as single-shot instructions and need to be split into separate steps; citation attribution isn't claim verification. ✅ published
6. **Evaluating the whole pipeline** — recall@k shows reranking actually *hurts* overall accuracy on this query set; an LLM-as-judge groundedness check has its own distinct blind spots. ✅ published — last of the originally planned 6 parts.
7. **Chunking** — chunk size, overlap, and fixed-size vs. sentence-aware strategy, measured against a fixture long enough to need splitting at all. ✅ published
8. **Vector search at scale** — brute force vs. HNSW (approximate nearest-neighbor) at up to 10,000 documents, plus pre- vs. post-filter metadata filtering. ✅ published
9. **Multi-hop retrieval** — query decomposition tested against naive single-shot retrieval on fact-chain questions; decomposition didn't win. ✅ published
10. **RAG vs. long-context LLMs** — the real token cost and latency of stuffing vs. retrieving, and a direct (negative) test for "lost in the middle" at this scale. ✅ published
11. (possible stretch) Query rewriting/expansion, NLI-based groundedness checking

**Companion piece (not a numbered part):** [RAG Interview Questions](https://dineshreddythimmasani.com/blog/rag-interview-questions/) — easy to senior level, every answer backed by a specific number or failure case from this series rather than a generic textbook answer.

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

python3 eval_retrieval.py     # Part 6, recall@k (+ precision@k/MRR/NDCG) across every retrieval stage
python3 eval_generation.py    # Part 6, LLM-as-judge groundedness evaluation

python3 freshness.py          # Part 1 addition, index freshness (BM25 vs embeddings)
python3 prompt_injection.py   # Part 5 addition, prompt injection via retrieved content

python3 eval_chunking.py      # Part 7, chunking: boundary splits, chunk-size sweep, strategy comparison

pip install hnswlib           # Part 8 only (already in requirements.txt)
python3 eval_ann.py           # Part 8, brute force vs HNSW, pre- vs post-filter metadata filtering

python3 eval_multihop.py      # Part 9, single-shot vs decomposed retrieval on fact-chain questions

python3 eval_longcontext.py   # Part 10, RAG-vs-stuffing cost table, lost-in-the-middle test
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
