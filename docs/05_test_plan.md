# Test Plan

**Author (persona):** QA / Evaluation Engineer

## 1. Test Strategy

| Level | Scope | Tooling | Location |
|---|---|---|---|
| Unit | Chunking, hybrid score fusion, ETL transform functions, cloud provider interface contracts | `pytest` | `tests/unit/` |
| Integration | API endpoints end-to-end against an in-memory vector store | `pytest` + `httpx`/`fastapi.testclient` | `tests/integration/` |
| GenAI Quality (offline eval) | Retrieval quality (hit-rate@k, MRR) for dense-only vs. lexical-only vs. hybrid; RAG answer groundedness spot-check | Custom harness | `tests/evaluation/` |
| Static analysis | Lint + type-check | `ruff`, `mypy` (best-effort) | CI pipeline |
| Load (design only, not executed here) | Concurrent query throughput | `locust` script provided | `tests/integration/load/` |

## 2. Test Case Matrix (representative)

| ID | Area | Given | When | Then |
|---|---|---|---|---|
| UT-01 | Chunking | A document of 2000 chars, chunk_size=500, overlap=50 | `chunk_text()` runs | Returns ceil-consistent chunk count, each chunk ≤ chunk_size, adjacent chunks overlap by 50 chars |
| UT-02 | Hybrid fusion | Dense scores and BM25 scores for 3 candidates | `fuse_scores(alpha=0.5)` runs | Fused score = 0.5*dense + 0.5*bm25 for every candidate, ranking is stable-sorted descending |
| UT-03 | ETL transform | Raw text with extra whitespace and control chars | `clean_text()` runs | Whitespace collapsed, control chars stripped, original semantic content preserved |
| UT-04 | Cloud provider interface | Any concrete provider (AWS/Azure/GCP/local) | `.upload()`/`.download()` called | All providers satisfy the same abstract interface (structural test via `issubclass`/duck typing) |
| IT-01 | API /health | Service started | `GET /health` | 200, `status: ok` |
| IT-02 | API /ingest | Sample doc present | `POST /ingest` | 200, `chunks_indexed > 0`, chunks appear in vector store |
| IT-03 | API /query | Corpus ingested | `POST /query` with in-corpus question | 200, non-empty `answer`, ≥1 citation, citation chunk_id exists in store |
| IT-04 | API /agent/run | Corpus ingested | `POST /agent/run` with multi-part goal | 200, `steps_used ≤ max_steps`, final step action == finish |
| IT-05 | Auth | No `X-API-Key` header | `POST /query` | 401 with structured error body |
| EVAL-01 | Retrieval quality | Labeled query→expected-chunk pairs | Run dense-only, lexical-only, hybrid retrieval | Hybrid hit-rate@5 ≥ max(dense-only, lexical-only) on the eval set (report in `11_rag_evaluation_report.md`) |

## 3. Entry / Exit Criteria

- **Entry:** ETL pipeline runs clean on sample corpus; API boots locally.
- **Exit:** All unit + integration tests pass in CI; evaluation harness produces a report; no P0/P1 defects open.

## 4. Defect Severity Definitions

| Severity | Definition |
|---|---|
| P0 | Data loss, security exposure, or pipeline cannot ingest/query at all |
| P1 | Incorrect answers/citations, agent infinite loop, API 5xx on valid input |
| P2 | Degraded quality (e.g., poor ranking) not blocking core flow |
| P3 | Cosmetic / documentation |

## 5. Evaluation Metrics Definitions

- **Hit-rate@k** — fraction of eval queries where the expected chunk appears in the top-k retrieved.
- **MRR (Mean Reciprocal Rank)** — average of 1/rank of the first relevant chunk.
- **Groundedness spot-check** — manual review that the generated answer only asserts facts present in the cited chunks.
