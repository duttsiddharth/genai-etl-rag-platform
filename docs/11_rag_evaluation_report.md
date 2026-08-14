# RAG Evaluation Report — Hybrid Retrieval Optimization

**Author (persona):** QA / Evaluation Engineer, with GenAI Developer
**Harness:** `tests/evaluation/test_retrieval_quality.py`
**Raw output:** `artifacts/eval/retrieval_quality_report.json` (regenerated on every test run — the numbers below are a captured run)

## 1. Method

An 8-query labeled evaluation set (`EVAL_QUERIES` in the harness) pairs each natural-language question with the sample-corpus document it should be grounded in. The sample corpus (`data/sample_docs/`) was chunked with the default recursive strategy (400 chars, 40 overlap) and embedded with the dependency-free hashing embedding provider, then indexed into both a dense vector index and a BM25 lexical index. For each query, retrieval was run three ways — dense-only (`alpha=1.0`), lexical-only (`alpha=0.0`), and the hybrid default (`alpha=0.6`) — and scored with hit-rate@1, hit-rate@5, and MRR (mean reciprocal rank of the first correctly-sourced chunk).

## 2. Results

| Strategy | alpha | Hit-rate@1 | Hit-rate@5 | MRR |
|---|---|---|---|---|
| Dense-only | 1.0 | 0.625 | 0.875 | 0.750 |
| Lexical-only (BM25) | 0.0 | 0.750 | 0.875 | 0.792 |
| **Hybrid (default)** | **0.6** | **0.750** | **0.875** | **0.8125** |

## 3. Alpha Sweep (hit-rate@5 / MRR)

| alpha | Hit-rate@5 | MRR |
|---|---|---|
| 0.0 | 0.875 | 0.7917 |
| 0.2 | 0.875 | 0.8125 |
| 0.4 | 0.875 | 0.8125 |
| 0.6 | 0.875 | 0.8125 |
| 0.8 | 0.875 | 0.7188 |
| 1.0 | 0.875 | 0.7500 |

## 4. Interpretation

On this small evaluation set, hit-rate@5 is saturated at 0.875 across every strategy — the corpus is small enough (5 documents) that the correct source almost always appears somewhere in the top 5 candidates regardless of retrieval method. **MRR is the more informative metric here**, because it captures *where* the correct chunk lands in the ranking, not just whether it appears at all. On that measure:

- **Dense-only retrieval under-ranks lexically-distinctive queries.** Questions referencing exact identifiers (ticket IDs like "TCK-1042", specific config names like "max_steps") are where dense-only embeddings blur relevance, dragging MRR down to 0.750.
- **Lexical-only retrieval is a strong baseline on this corpus** (MRR 0.792) precisely because the corpus vocabulary is narrow and technical, which favors exact term matching.
- **Hybrid retrieval (alpha=0.6) achieves the best MRR (0.8125)** in the sweep, matching the plateau also reached at alpha=0.2 and alpha=0.4 — i.e., any moderate blend of dense and lexical signal outperforms either pure strategy, with returns flattening as alpha approaches 0.4–0.6. This validates the architecture's default alpha (`docs/03_architecture_design.md` §6) and directly reflects a real incident captured in the sample corpus itself (ticket TCK-1055, where an all-dense default under-served SKU-heavy queries and was fixed by lowering alpha).

## 5. Threats to Validity

- The evaluation corpus is intentionally small (5 documents, 8 labeled queries) for a fast, dependency-free, CI-runnable demonstration; hit-rate@5 saturating at 0.875 (7/8) reflects that the harness currently has one query the retriever does not surface correctly at k=5 even at the best alpha — a candidate for the next iteration's query-set expansion.
- The hashing embedding provider is a lexical-hash-based vectorizer, not a neural semantic embedding model, so "dense-only" here approximates — but does not fully represent — the ceiling a neural embedding model (`local-sentence-transformers`, OpenAI, Bedrock Titan, Azure OpenAI, or Vertex AI embeddings) would reach on semantic paraphrase queries. The interface swap to a neural provider is a one-line config change (`EMBEDDING_PROVIDER=local-sentence-transformers`); re-running this same harness after that swap is the recommended next validation step.

## 6. Recommendation

Ship `alpha=0.6` as the platform default (already reflected in `src/api/state.py`), keep `alpha` exposed as a per-request override in `/query` (already implemented), and grow the labeled evaluation set as the corpus grows in production so hit-rate@5 stops being a saturated metric and starts being a useful regression signal — tracked via the `retrieval_hit_rate` Prometheus gauge and the weekly re-run cadence described in `docs/06_mlops_devops_plan.md` §7.
