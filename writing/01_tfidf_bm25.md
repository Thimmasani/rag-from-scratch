# Building RAG From Scratch, Part 1: TF-IDF, BM25, and Why BM25 Still Wins

*Series: Building RAG From Scratch. Part 1 of ~6 — this one covers lexical retrieval. Part 2 adds embeddings and shows exactly where lexical search breaks.*

---

Every RAG tutorial starts with embeddings. Chunk your docs, embed them, throw them in a vector DB, done. What almost nobody shows you is the retrieval method that came *before* embeddings and that production search systems still haven't retired: BM25.

Google "car" and a page about "automobiles" won't match on vocabulary alone. That's the core limitation lexical search has always had — and the reason semantic search exists. But lexical search solves a different problem better than embeddings ever will: finding a rare exact string, a product code, a name, an ID. If your RAG system can't reliably retrieve "SKU-48213-B" because it only has a dense retriever, that's not a hypothetical failure, it's a Tuesday.

So before building the embedding side of RAG, let's build the lexical side by hand: TF-IDF first, then BM25, then a head-to-head comparison showing exactly where BM25 wins and why.

All code below is plain Python, no `sklearn`, no `rank_bm25`. The point of a from-scratch series is that you can verify every number by hand.

---

## The toy corpus

Six short documents, designed so every score is hand-checkable:

```python
DOCUMENTS = [
    "the car needs an oil change and new brake pads",                      # doc0
    "my automobile requires fresh engine oil and brake service",           # doc1
    "replacement part SKU-48213-B is on back order until next week",       # doc2
    "the weather today is sunny with a light breeze in the afternoon",     # doc3
    "the car dealership offers free oil change on the weekend",            # doc4
    "oil oil oil best oil change deals oil discount oil coupon oil near "
    "me oil prices oil service oil special oil offer cheap oil oil shop",  # doc5
]
```

doc0 and doc1 are the same topic (car maintenance) written with different vocabulary — the vocabulary mismatch case lexical search struggles with. doc2 has a one-off exact code. doc3 is an unrelated distractor. doc4 overlaps doc0's vocabulary so IDF has something to discriminate on. doc5 is deliberately spammy: it repeats "oil" eleven times with barely any real content, there to expose TF-IDF's biggest weakness later in this post.

---

## TF-IDF: term frequency, weighted by rarity

The idea behind TF-IDF is simple: a term that appears often *in this document* is probably important to it (term frequency), but a term that appears in *every* document in the corpus isn't telling you anything useful (hence inverse document frequency to downweight it).

**Term frequency (TF)** — how many times term *t* appears in document *d*. Just a count.

**Inverse document frequency (IDF)** — how rare is term *t* across the whole corpus. The smoothed version (what scikit-learn uses by default):

```
idf(t) = ln( (1 + N) / (1 + df(t)) ) + 1
```

- `N` = total number of documents
- `df(t)` = number of documents containing term *t*
- The `+1`s in numerator and denominator prevent division by zero and keep rare terms from exploding to infinity
- The trailing `+ 1` keeps every term's weight at least 1, so even a term appearing in all documents still contributes something

A term in every document gets `idf ≈ 1` (low weight). A term in only one document out of six gets a much higher weight. That's the whole mechanism: common words get downweighted, rare words get upweighted.

**Why `ln` specifically?** Rarity should discount gently, not linearly — going from "in 1000 docs" to "in 500 docs" should matter less than going from "in 2 docs" to "in 1 doc." Log compresses large ratios and keeps IDF from blowing up or swinging wildly as corpus size changes.

**TF-IDF weight** for a term in a document is just:

```
tfidf(t, d) = tf(t, d) * idf(t)
```

Each document becomes a vector over the whole vocabulary, where each coordinate is that term's TF-IDF weight (0 if the term doesn't appear). To rank documents against a query, embed the query the same way and compute **cosine similarity**:

```
cos(q, d) = (q · d) / (||q|| * ||d||)
```

Cosine similarity ignores document length (it normalizes both vectors to unit length), which solves one problem but, as we'll see, creates another.

### Worked example on the real corpus

Our corpus has `N = 6` documents. Let's compute IDF for a few terms by hand:

| term | appears in (df) | idf = ln((1+6)/(1+df)) + 1 |
|---|---|---|
| `the` | 3 docs (doc0, doc3, doc4) | ln(7/4) + 1 = **1.5596** |
| `car` | 2 docs (doc0, doc4) | ln(7/3) + 1 = **1.8473** |
| `oil` | 4 docs (doc0, doc1, doc4, doc5) | ln(7/5) + 1 = **1.3365** |
| `automobile` | 1 doc (doc1 only) | ln(7/2) + 1 = **2.2528** |

Notice the pattern: `automobile` appears in only one document and gets the highest weight (2.2528). `oil` appears in four of six documents (it's common in this corpus) and gets downweighted to 1.3365, even though it's clearly an important word, because it doesn't help *discriminate* between documents.

Now vectorize doc0 (`"the car needs an oil change and new brake pads"`, 10 tokens, each appearing once so `tf = 1` for all):

```
doc0 vector = {
  the: 1 * 1.5596 = 1.5596,  car: 1 * 1.8473 = 1.8473,
  oil: 1 * 1.3365 = 1.3365,  change: 1 * 1.5596 = 1.5596,
  needs: 2.2528, an: 2.2528, and: 1.8473, new: 2.2528,
  brake: 1.8473, pads: 2.2528
}
```

Query `"car oil change"` vectorizes to `{car: 1.8473, oil: 1.3365, change: 1.5596}` (each term appears once in the query too).

Cosine similarity only sums over terms that appear in *both* vectors:

```
dot(q, doc0) = (1.8473 * 1.8473) + (1.3365 * 1.3365) + (1.5596 * 1.5596)
             = 3.4125 + 1.7862 + 2.4324 = 7.6311

||q||    = sqrt(1.8473² + 1.3365² + 1.5596²) = sqrt(7.6311) = 2.7624
||doc0|| = sqrt(sum of all 10 squared weights in doc0) = 6.0982

cos(q, doc0) = 7.6311 / (2.7624 * 6.0982) = 0.4530
```

That matches the `0.4530` TF-IDF score for doc0 on this query that the code prints (see the full table further down — numbers shifted slightly from earlier drafts of this post after adding doc5 to the corpus, since IDF depends on the whole corpus).

### Run it yourself

The repo includes a small interactive CLI so you don't have to take any of this on faith — type your own query against the same corpus and watch both scorers respond live:

```bash
cd retrieval
python3 query.py
> car oil change
> automobile
> quit
```

It prints TF-IDF and BM25 scores side by side for whatever you type, using the exact same `TfidfRetriever` and `Bm25Retriever` classes shown in this post.

### The code

```python
import math
import re
from collections import Counter

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())

class TfidfRetriever:
    def __init__(self, documents: list[str]):
        self.documents = documents
        self.tokenized_docs = [tokenize(doc) for doc in documents]
        self.n_docs = len(documents)

        vocab_set = set()
        for tokens in self.tokenized_docs:
            vocab_set.update(tokens)
        self.vocab = sorted(vocab_set)

        self.idf = self._compute_idf()
        self.doc_vectors = [self._vectorize(tokens) for tokens in self.tokenized_docs]

    def _compute_idf(self) -> dict[str, float]:
        df = Counter()
        for tokens in self.tokenized_docs:
            for term in set(tokens):
                df[term] += 1
        return {
            term: math.log((1 + self.n_docs) / (1 + df[term])) + 1
            for term in self.vocab
        }

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        tf = Counter(tokens)
        return {t: c * self.idf[t] for t, c in tf.items() if t in self.idf}

    @staticmethod
    def _cosine_sim(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
        shared = vec_a.keys() & vec_b.keys()
        dot = sum(vec_a[t] * vec_b[t] for t in shared)
        norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
        norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        query_vec = self._vectorize(tokenize(query))
        scores = [(i, self._cosine_sim(query_vec, dv)) for i, dv in enumerate(self.doc_vectors)]
        scores.sort(key=lambda p: p[1], reverse=True)
        return scores[:top_k]
```

Note the tokenizer keeps hyphenated tokens intact (`[a-z0-9]+(?:-[a-z0-9]+)*`) — that's deliberate, so `SKU-48213-B` survives as one token instead of being split into `sku`, `48213`, `b`. That choice matters later.

---

## Where TF-IDF falls short

TF-IDF has two structural weaknesses that show up as soon as you test it:

**1. No term frequency saturation.** If "oil" appears once, TF-IDF gives it weight `1 * idf`. If it appears 10 times, weight is `10 * idf` — linear. But a document repeating a word 10 times isn't 10x more relevant than one that mentions it once. Relevance should saturate.

**2. No native document length normalization in the raw score** — cosine similarity handles this by normalizing vectors to unit length, but that's a global fix applied after the fact, not something the scoring function itself accounts for term-by-term. This matters once you want finer control (which is exactly what BM25 adds).

This is where BM25 comes in. BM25 isn't a totally different idea, it's TF-IDF with both of these problems fixed directly in the scoring formula.

---

## BM25: TF-IDF with saturation and length normalization

BM25 ("Best Match 25") comes from the Okapi information retrieval system in the 1980s-90s and is still the default scorer in Elasticsearch and Lucene today. The formula, per query term:

```
score(d, q) = Σ IDF(t) * [ f(t,d) * (k1 + 1) ] / [ f(t,d) + k1 * (1 - b + b * |d| / avgdl) ]
```

- `f(t, d)` — raw term frequency of *t* in document *d*
- `|d|` — length of document *d* (token count)
- `avgdl` — average document length across the corpus
- `k1` (typically 1.2–2.0) — controls how quickly TF saturates
- `b` (typically 0.75) — controls how strongly document length is penalized

BM25's IDF variant is also slightly different from TF-IDF's:

```
IDF(t) = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )
```

### Why the denominator fixes saturation

Look at just the TF part: `f * (k1+1) / (f + k1 * length_norm)`. As `f` grows, this expression approaches `k1 + 1` asymptotically — it flattens out. Going from 1 occurrence to 2 gives a meaningful jump; going from 10 to 11 gives almost nothing. That's saturation: diminishing returns on repeated terms, matching real relevance judgments much better than TF-IDF's linear scaling.

### Why `length_norm` fixes document length

`length_norm = 1 - b + b * (|d| / avgdl)`. If a document is exactly average length, `length_norm = 1` and nothing changes. If it's longer than average, `length_norm > 1`, which inflates the denominator and *lowers* the score — term matches in a bloated document are discounted, because a long document matching a term by sheer size isn't as strong a relevance signal as a short, focused document matching the same term.

### Worked example: BM25 score for doc0, query "car oil change"

Corpus stats first: `N = 6` documents, document lengths are `[10, 9, 10, 12, 10, 26]` (doc5, the keyword-stuffed one, is the longest at 26 tokens), so:

```
avgdl = (10+9+10+12+10+26) / 6 = 12.8333
```

For doc0 (`|d| = 10` tokens):

```
length_norm = 1 - 0.75 + 0.75 * (10 / 12.8333) = 0.8344
```

With `k1 = 1.5`, score each query term against doc0 (every term here has `f(t, doc0) = 1`, it appears once):

| term | df | IDF = ln((N-df+0.5)/(df+0.5)+1) | f(t,doc0) | contribution |
|---|---|---|---|---|
| `car` | 2 | ln((6-2+0.5)/(2+0.5)+1) = **1.0296** | 1 | 1.0296 × (1×2.5)/(1+1.5×0.8344) = **1.1432** |
| `oil` | 4 | ln((6-4+0.5)/(4+0.5)+1) = **0.4418** | 1 | 0.4418 × 2.5/2.2516 = **0.4906** |
| `change` | 3 | ln((6-3+0.5)/(3+0.5)+1) = **0.6931** | 1 | 0.6931 × 2.5/2.2516 = **0.7696** |

```
BM25(doc0, "car oil change") = 1.1432 + 0.4906 + 0.7696 = 2.4034
```

That's the exact `2.4034` the code prints for doc0 on this query. Note `oil` contributes the least of the three terms here (0.4906) because it has the lowest IDF — it appears in 4 of 6 documents in this corpus, so it's less discriminating than `car` or `change`. Same underlying idea as TF-IDF's IDF term, just with BM25's slightly different smoothing.

### The code

```python
import math
from collections import Counter
from tfidf import tokenize  # same tokenizer, for a fair comparison

class Bm25Retriever:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.tokenized_docs = [tokenize(doc) for doc in documents]
        self.n_docs = len(documents)
        self.doc_lengths = [len(toks) for toks in self.tokenized_docs]
        self.avg_doc_length = sum(self.doc_lengths) / self.n_docs
        self.doc_term_freqs = [Counter(toks) for toks in self.tokenized_docs]

        self.doc_freq = Counter()
        for tokens in self.tokenized_docs:
            for term in set(tokens):
                self.doc_freq[term] += 1

        self.idf = {
            term: math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
            for term, df in self.doc_freq.items()
        }

    def _score(self, query_tokens: list[str], doc_idx: int) -> float:
        score = 0.0
        doc_len = self.doc_lengths[doc_idx]
        term_freqs = self.doc_term_freqs[doc_idx]
        length_norm = 1 - self.b + self.b * (doc_len / self.avg_doc_length)

        for term in query_tokens:
            f = term_freqs.get(term, 0)
            if f == 0 or term not in self.idf:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * length_norm
            score += self.idf[term] * (numerator / denominator)
        return score

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        query_tokens = tokenize(query)
        scores = [(i, self._score(query_tokens, i)) for i in range(self.n_docs)]
        scores.sort(key=lambda p: p[1], reverse=True)
        return scores[:top_k]
```

---

## Head-to-head: where BM25 actually changes the ranking

The vocabulary-mismatch queries from earlier show BM25 and TF-IDF *agreeing* on ranking (both fail the same way on "car" vs "automobile"). To see BM25 actually win, we need a case that exercises the thing TF-IDF structurally lacks: saturation. That's what doc5 is for.

**Query: `"car oil change"`** — doc0 (natural, concise, mentions each term once) vs doc5 (keyword-stuffed, repeats "oil" eleven times):

| doc | TF-IDF | BM25 | rank TF-IDF | rank BM25 |
|---|---|---|---|---|
| doc5: "oil" × 11, keyword-stuffed | **0.4870** | 1.3915 | **#1** | #3 |
| doc0: car/oil/brakes, natural phrasing | 0.4530 | **2.4034** | #2 | **#1** |
| doc4: car/oil, overlaps doc0 | 0.4310 | 2.4034 | #3 | #1 (tie) |

**This is the rank flip.** TF-IDF puts the spammy doc5 in first place, ahead of doc0, purely because repeating "oil" eleven times linearly inflates its TF-IDF weight for that term. BM25 correctly demotes doc5 to last, because `f=11` for "oil" barely scores higher than `f=1` once it's run through BM25's saturating denominator — repeating a word 11 times stops paying off almost immediately.

To make the saturation effect even starker, strip it down to a single-term query:

**Query: `"oil change"`** (doc5 vs doc0, isolating just the repeated term):

| doc | TF-IDF | BM25 |
|---|---|---|
| doc5: "oil" × 11 | **0.6551** | 1.3915 |
| doc0: "oil" × 1, natural | 0.3368 | 1.2602 |

TF-IDF gives doc5 nearly **2x** the score of doc0 (0.6551 vs 0.3368) for repeating one word eleven times instead of once. BM25 narrows that gap to barely a 10% difference (1.3915 vs 1.2602). This is term-frequency saturation made visible: BM25's `k1` parameter caps how much repetition can buy a document, which is exactly the behavior you want, since "mentions oil 11 times" is a terrible proxy for "is actually about oil changes."

This is also a direct, hands-on preview of why production search ranking cares about this at all: keyword-stuffed pages are a known adversarial pattern against naive TF scoring, and BM25's saturation is a big part of why it, not raw TF-IDF, became the industry default scorer.

### The other three queries: vocabulary mismatch and coincidental overlap

The remaining queries from the corpus show cases where BM25 and TF-IDF *agree* — useful for understanding what lexical search as a category can't fix, regardless of which scorer you pick.

**Query: `"automobile brake service"`** — mirrors the car/automobile mismatch in the other direction; doc1 (says "automobile") wins both methods, doc0 (says "car") ranks far behind despite being the same topic.

**Query: `"SKU-48213-B"`** — the case lexical search was built for. The exact code is a single rare token (`df=1`, high IDF), both methods immediately and correctly rank doc2 first. A purely embedding-based retriever has no guarantee of surfacing this at all — "SKU-48213-B" has no learned semantic meaning, it's just a string a model has likely never seen.

**Query: `"weekend weather"`** — doc4 (car dealership, mentions "weekend") outranks doc3 (actually about weather) in both methods, because "weekend" and "weather" each appear in exactly one document (same IDF) and doc4 happens to be slightly shorter, giving it a small length-normalization edge in BM25's case. Neither method understands "weekend weather" as a semantic unit; they're independently matching two unrelated single words. Call this "coincidental lexical overlap" — it's vocabulary mismatch's cousin, and no amount of `k1`/`b` tuning fixes it, because the problem is the premise (matching tokens, not meaning), not the scoring formula.

---

## So why does BM25 win, concretely

- **TF saturation matters once documents repeat terms heavily.** The doc5 example above isn't a toy edge case — keyword stuffing is a real, adversarial pattern against naive TF scoring. TF-IDF's linear scaling lets repetition dominate; BM25's saturation (via `k1`) caps the damage, which is why doc5 drops from TF-IDF's #1 to BM25's #3 on the same query.
- **Length normalization matters once your corpus has mixed document lengths.** A long document matching a query term by sheer surface area isn't automatically more relevant than a short, focused one. TF-IDF's cosine normalization is a global, after-the-fact fix; BM25 bakes it into the per-term scoring directly via `b`, which behaves more predictably as corpora grow and document lengths vary.
- **BM25 is the actual industry default.** Elasticsearch, Lucene (and therefore Solr), and most production search engines ship BM25, not raw TF-IDF, as their lexical scorer. If you're building hybrid retrieval for RAG, BM25 is almost always the lexical half you'll reach for.

The deeper point for a RAG system: neither TF-IDF nor BM25 understands meaning at all. Both are entirely surface-token matching. They will reliably win on exact codes, names, and IDs (the SKU example), and they will reliably lose on paraphrase and vocabulary mismatch ("car" vs "automobile") or coincidental overlap ("weekend weather"). That's not a flaw to engineer away, it's a fundamental property of lexical matching, and it's exactly why production RAG systems pair BM25 with semantic retrieval instead of picking one.

---

## What's next

Part 2 (`writing/02_embeddings.md`) takes the exact same toy corpus and queries and runs them through sentence embeddings instead, so you can see "car" vs "automobile" and "weekend weather" resolve correctly once meaning enters the picture — plus a dedicated experiment on what actually happens to an exact code like `SKU-48213-B` once lexical matching is gone (the real answer is more interesting than a simple failure).

Full code for this post: `ai_projects/rag_from_scratch/retrieval/` — `tfidf.py`, `bm25.py`, `compare.py`, and `query.py` if you want to try your own queries against the corpus.

If you've hit the vocabulary mismatch problem or wondered how embeddings handle exact codes in a real RAG system, check out part 2 for a concrete answer.

---

**Tags:** `Machine Learning`, `Artificial Intelligence`, `NLP`, `Information Retrieval`, `RAG`
