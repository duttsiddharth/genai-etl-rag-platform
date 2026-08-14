# Solution Architecture Design

**Author (persona):** Solution Architect
**Status:** Approved for build

## 1. Context Diagram

```mermaid
flowchart LR
    subgraph Sources
        A1[PDFs]
        A2[HTML pages]
        A3[Ticket exports JSON]
    end
    A1 & A2 & A3 --> ETL[ETL Pipeline\nsrc/etl]
    ETL --> Chunk[Chunking\nsrc/genai/chunking.py]
    Chunk --> Embed[Embeddings\nsrc/genai/embeddings.py]
    Embed --> VDB[(Vector Store\nChroma/FAISS)]
    Embed --> BM25[(BM25 Lexical Index)]
    VDB --> Hybrid[Hybrid Retriever]
    BM25 --> Hybrid
    Hybrid --> RAG[RAG Chain]
    RAG --> API[FastAPI Service]
    Agent[Agent Orchestrator] --> Hybrid
    Agent --> Tools[Tools: Calculator, WebLookup stub]
    API --> Agent
    API --> Clients[Internal Apps / Chat UI]
    API --> Cloud[Cloud Provider Layer\nAWS/Azure/GCP]
    Cloud --> S3[(Object Storage)]
    Cloud --> LLMApi[(Managed LLM API\nBedrock/Azure OpenAI/Vertex)]
    API --> Mon[Monitoring & Logging]
```

## 2. Component Responsibilities

| Component | Responsibility | Key files |
|---|---|---|
| ETL Pipeline | Extract raw text + metadata, normalize/clean, load into a staging store and trigger chunk/embed | `src/etl/extract.py`, `transform.py`, `load.py`, `pipeline.py` |
| Chunking | Split documents into retrieval-sized units with configurable strategy and overlap | `src/genai/chunking.py` |
| Embeddings | Turn chunks/queries into vectors via a pluggable provider (local or cloud) | `src/genai/embeddings.py` |
| Vector Store | Persist vectors + metadata, similarity search (cosine) | `src/genai/vector_store.py` |
| Hybrid Retriever | Fuse dense similarity + BM25 lexical scores (weighted, `alpha` tunable), optional re-rank | `src/genai/hybrid_retriever.py` |
| RAG Chain | Build grounded prompt from retrieved chunks, call LLM, return answer + citations | `src/genai/rag_chain.py` |
| Agent Orchestrator | Plan → Act → Observe loop; selects tools (retrieval, calculator); step-capped for cost control | `src/genai/agents/orchestrator.py`, `tools.py` |
| Fine-tuning module | LoRA-style domain adaptation demo on a small model/embedding set | `src/genai/finetuning/lora_finetune_demo.py` |
| API Layer | REST endpoints, request validation, auth stub, error handling | `src/api/*` |
| Cloud Provider Layer | Abstracts storage + managed-model access across AWS/Azure/GCP | `src/cloud/*` |
| Monitoring | Structured logging + Prometheus metrics | `src/monitoring/*` |

## 3. Data Flow — Ingestion (ETL)

```mermaid
sequenceDiagram
    participant Src as Source Docs
    participant Ext as extract.py
    participant Tr as transform.py
    participant Ld as load.py
    participant Ch as chunking.py
    participant Em as embeddings.py
    participant VS as VectorStore

    Src->>Ext: read file (pdf/html/txt/json)
    Ext->>Tr: raw text + metadata
    Tr->>Tr: clean, normalize, dedupe, PII-scrub hook
    Tr->>Ld: normalized Document
    Ld->>Ld: write manifest record (id, checksum, source)
    Ld->>Ch: Document
    Ch->>Em: List[Chunk]
    Em->>VS: List[(chunk_id, vector, metadata)]
    VS-->>Ld: ack (count indexed)
```

## 4. Data Flow — Hybrid RAG Query

```mermaid
sequenceDiagram
    participant U as Client
    participant API as FastAPI /query
    participant HR as HybridRetriever
    participant VS as VectorStore (dense)
    participant BM as BM25Index (lexical)
    participant RC as RAGChain
    participant LLM as LLM Provider

    U->>API: POST /query {question}
    API->>HR: retrieve(question, k)
    HR->>VS: dense_search(question_vec, k*2)
    HR->>BM: lexical_search(question, k*2)
    VS-->>HR: dense candidates + scores
    BM-->>HR: lexical candidates + scores
    HR->>HR: fuse scores (alpha * dense + (1-alpha) * lexical), top-k
    HR-->>RC: ranked chunks
    RC->>LLM: grounded prompt (context + question)
    LLM-->>RC: answer
    RC-->>API: answer + citations + retrieval scores
    API-->>U: JSON response
```

## 5. Agentic Workflow Orchestration

```mermaid
sequenceDiagram
    participant U as Client
    participant API as FastAPI /agent/run
    participant Ag as Orchestrator
    participant HR as HybridRetriever (tool)
    participant Calc as Calculator (tool)
    participant LLM as Planner LLM

    U->>API: POST /agent/run {goal}
    API->>Ag: run(goal)
    loop until done or max_steps
        Ag->>LLM: plan next step given trace
        LLM-->>Ag: {action: retrieve|calculate|finish, args}
        alt action == retrieve
            Ag->>HR: retrieve(args.query)
            HR-->>Ag: chunks
        else action == calculate
            Ag->>Calc: evaluate(args.expression)
            Calc-->>Ag: result
        else action == finish
            Ag-->>API: final answer + full trace
        end
    end
    API-->>U: {answer, trace[]}
```

## 6. Technology Stack

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.11 | Type-hinted, `pyproject.toml` packaging |
| API framework | FastAPI + Uvicorn | Async, OpenAPI auto-docs |
| Embeddings | `sentence-transformers` (local default) / OpenAI / Bedrock Titan / Azure OpenAI / Vertex AI | Pluggable via `EMBEDDING_PROVIDER` env var |
| Vector store | Chroma (local, default) / FAISS | Pluggable via `VECTOR_STORE` env var; interface supports adding pgvector/OpenSearch/Pinecone |
| Lexical index | `rank-bm25` | Powers hybrid fusion |
| LLM | Local deterministic stub (default, offline-safe) / OpenAI / Bedrock / Azure OpenAI / Vertex AI | Pluggable via `LLM_PROVIDER` |
| Fine-tuning | HuggingFace `transformers` + `peft` (LoRA) demo | Small model, CPU-runnable subset |
| Containerization | Docker, multi-stage build | `infra/docker/Dockerfile` |
| Orchestration | Kubernetes (Deployment, Service, HPA, ConfigMap) | `infra/k8s/` |
| IaC | Terraform (AWS: S3 bucket, ECR repo, IAM role, CloudWatch log group) | `infra/terraform/` |
| CI/CD | GitHub Actions | Lint → test → build → (push image) |
| Monitoring | `prometheus-client` metrics + structured JSON logs | `src/monitoring/` |

## 7. Cloud Deployment Topology (AWS primary)

```mermaid
flowchart TB
    subgraph AWS
        ALB[Application Load Balancer]
        subgraph EKS[EKS Cluster]
            Pod1[API Pod]
            Pod2[API Pod]
            HPA[HPA Controller]
        end
        S3[(S3 - raw docs + manifests)]
        ECR[(ECR - container images)]
        CW[CloudWatch Logs/Metrics]
        Bedrock[(Bedrock - LLM + embeddings)]
        IAM[IAM Roles - least privilege]
    end
    Client-->ALB-->Pod1
    ALB-->Pod2
    HPA-.scales.->Pod1
    HPA-.scales.->Pod2
    Pod1-->S3
    Pod1-->Bedrock
    Pod1-->CW
    ECR-.image pull.->Pod1
    IAM-.assumed by.->Pod1
```

Azure equivalent: AKS + Blob Storage + Azure Container Registry + Azure OpenAI + Azure Monitor.
GCP equivalent: GKE + GCS + Artifact Registry + Vertex AI + Cloud Monitoring.
The `src/cloud/` interface means only configuration and the Terraform/IaC module change — application code does not.

## 8. Security Architecture

- Secrets via environment variables / cloud secret manager (never committed — see `.env.example`).
- API key auth stub on all mutating endpoints (`X-API-Key` header), designed to be swapped for OAuth2/OIDC in production.
- Least-privilege IAM role for the workload (Terraform module scopes S3 + Bedrock invoke only).
- PII-scrub hook in `transform.py` (regex-based redaction pluggable point) before data reaches the vector store.
- Network: ALB → private subnet pods; vector store not internet-exposed.

## 9. Scalability & Cost Considerations

- Stateless API pods scale horizontally behind HPA on CPU + custom retrieval-latency metric.
- Vector store supports incremental upsert so re-indexing the whole corpus is not required per ingestion.
- Agent loop has a hard `max_steps` and per-run token budget to bound cost.
- Embedding calls are batched; local embedding model avoids per-call API cost during development.
