# Business & Functional Requirements Document (BRD/FRD)

**Author (persona):** Business Analyst
**Reviewed by:** Product Owner, Solution Architect, GenAI Developer

## 1. Business Requirements

| ID | Requirement | Rationale |
|---|---|---|
| BR-1 | Employees must be able to ask natural-language questions and receive answers grounded in the internal document corpus | Reduce time-to-answer, reduce hallucination risk vs. raw LLM |
| BR-2 | The system must support documents in multiple formats without manual reformatting | Corpus is heterogeneous (PDF, HTML, exported tickets) |
| BR-3 | The system must be auditable — every answer traceable to source chunks | Compliance / trust |
| BR-4 | The system must run on the organization's approved cloud (AWS primary) with a path to Azure/GCP | Avoid vendor lock-in, meet procurement policy |
| BR-5 | The platform must support increasingly complex, multi-step questions over time | Roadmap toward agentic research assistants |

## 2. Functional Requirements

| ID | Requirement | Priority |
|---|---|---|
| FR-1 | System shall extract text and metadata from PDF, HTML, TXT, and JSON sources | Must |
| FR-2 | System shall chunk documents using a configurable strategy (fixed, recursive, semantic) with overlap | Must |
| FR-3 | System shall embed chunks using a pluggable embedding provider (local sentence-transformers or cloud embedding API) | Must |
| FR-4 | System shall persist embeddings + metadata in a vector store supporting similarity search | Must |
| FR-5 | System shall support hybrid retrieval combining dense similarity and BM25 lexical search with a tunable fusion weight | Must |
| FR-6 | System shall generate answers via a RAG chain that cites the source chunk IDs used | Must |
| FR-7 | System shall expose ingest, query, and health operations via a documented REST API | Must |
| FR-8 | System shall support an agentic mode: given a complex query, the agent plans sub-steps, invokes tools (retrieval, calculator), and synthesizes a final answer with a visible reasoning trace | Must |
| FR-9 | System shall support swapping the underlying cloud provider (AWS/Azure/GCP) for storage and model access via configuration, not code changes | Must |
| FR-10 | System shall demonstrate a basic fine-tuning / domain-adaptation workflow and report before/after metrics | Must |
| FR-11 | System shall emit structured logs and Prometheus-compatible metrics for every request | Should |
| FR-12 | System shall be containerized and deployable to Kubernetes with horizontal autoscaling | Should |
| FR-13 | System shall run automated tests and linting on every push via CI/CD | Should |
| FR-14 | System shall provide a rollback-capable deployment strategy | Could |

## 3. Non-Functional Requirements

| ID | Category | Requirement |
|---|---|---|
| NFR-1 | Performance | P95 query latency < 3s for hybrid retrieval + generation on a corpus of ≤10k chunks (local model), documented separately for cloud LLM calls |
| NFR-2 | Scalability | Stateless API layer must scale horizontally; vector store must support incremental ingestion without full re-index |
| NFR-3 | Security | Secrets (API keys, cloud credentials) must never be hard-coded; loaded via environment/secret manager |
| NFR-4 | Reliability | API must return graceful, typed error responses; agent loop must have a max-step guard to prevent runaway loops/cost |
| NFR-5 | Observability | Every pipeline stage (extract, transform, load, embed, retrieve, generate) must emit a log line with duration and status |
| NFR-6 | Portability | No hard dependency on a single cloud SDK inside business logic — cloud access goes through an interface in `src/cloud/` |
| NFR-7 | Cost control | Agent loop and RAG chain must cap token usage / step count via configuration |

## 4. User Stories (sample, Gherkin acceptance criteria)

**US-1: Ingest a new document**
```
As a knowledge manager
I want to drop a PDF into the ingestion pipeline
So that its content becomes searchable

Given a PDF file in the configured source folder
When the ETL pipeline runs
Then the document is extracted, chunked, embedded, and stored in the vector store
And a load manifest entry records document id, chunk count, and checksum
```

**US-2: Ask a grounded question**
```
As an internal user
I want to ask a question via the API
So that I get an answer grounded in our documents with citations

Given the vector store contains ingested documents
When I POST a question to /query
Then I receive an answer, the source chunk ids, and a confidence/retrieval score
```

**US-3: Ask a multi-step research question**
```
As a power user
I want to ask a question that requires multiple lookups and a calculation
So that I get a synthesized answer instead of doing the steps myself

Given the agent orchestrator is enabled
When I POST a complex question to /agent/run
Then the response includes a step-by-step trace (plan, tool calls, observations) and a final synthesized answer
```

## 5. Constraints & Assumptions

- No production LLM API key is provisioned in this development workspace; the RAG/agent code defaults to a deterministic local "stub" LLM adapter so the pipeline is fully runnable offline, and swaps to OpenAI/Bedrock/Azure OpenAI/Vertex AI via one config flag (`LLM_PROVIDER`) when credentials are supplied.
- Embeddings default to a local `sentence-transformers` model to keep the reference implementation runnable without external calls; cloud embedding providers are supported via the same interface.
