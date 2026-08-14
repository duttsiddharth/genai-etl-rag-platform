# Project Charter — DocuMind AI: Enterprise Hybrid-RAG & Agentic Knowledge Platform

**Author (persona):** Product Owner / Engagement Sponsor
**Date:** 2026-02-02 (project kickoff, simulated)
**Engagement type:** 6-month contract, possible extension — mirrors the target GenAI Developer role (Toronto, ON — 4 days onsite)

## 1. Background

Internal teams (Support, Legal, Sales Engineering) spend significant time searching across scattered PDFs, Confluence pages, and ticket exports to answer domain questions. Answers are inconsistent because knowledge lives in unstructured documents that are never systematically ingested, chunked, or indexed. The organization wants a Generative AI platform that turns this unstructured corpus into a governed, queryable knowledge base with agentic workflows that can plan multi-step research tasks, not just answer single questions.

## 2. Objective

Design, build, and operationalize an end-to-end GenAI platform — **DocuMind AI** — that:

1. Extracts, transforms, and loads (ETL) heterogeneous documents into a governed knowledge store.
2. Implements chunking, embeddings, and a **hybrid retrieval** (semantic + lexical) RAG pipeline over a vector database.
3. Exposes GenAI capability through versioned APIs consumed by internal apps and a chat UI.
4. Adds **agentic orchestration** (planning, tool use, hybrid-RAG-aware retrieval optimization) for multi-step research tasks.
5. Runs on cloud infrastructure (AWS primary, portable to Azure/GCP) with full DevOps/MLOps practices: CI/CD, containerized deployment, model/pipeline monitoring, and logging.
6. Demonstrates basic fine-tuning / domain adaptation methods as an R&D spike, with a written experiment report.

This charter, and the artifacts that follow it, are built as a **portfolio-grade reference implementation** of everything listed in the target job description, so that every "Must Have" and "Nice to Have" line item is traceable to working code, a design decision, or a documented experiment. See `13_jd_traceability_matrix.md` for the full mapping.

## 3. In Scope

- Python ETL pipeline for document ingestion (PDF, HTML, TXT, JSON ticket exports).
- Chunking strategies (fixed-size, recursive, semantic) with configurable overlap.
- Vector store integration (Chroma / FAISS, abstracted so Pinecone/OpenSearch/pgvector can be swapped in).
- Hybrid retrieval (dense embeddings + BM25 lexical) with re-ranking and RAG answer generation.
- Agent orchestrator with tool-use (retrieval tool, calculator tool, web-lookup stub) and a planner/executor loop.
- REST API (FastAPI) for ingest, query, and agent-run operations.
- Cloud integration modules for AWS (S3, Bedrock), Azure (Blob Storage, Azure OpenAI), GCP (GCS, Vertex AI) behind a single provider-agnostic interface.
- IaC (Terraform for AWS), containerization (Docker), orchestration (Kubernetes manifests), CI/CD (GitHub Actions).
- Monitoring/logging: structured logs, Prometheus metrics, a model-quality/drift monitoring hook.
- Fine-tuning experiment (LoRA/PEFT-style adaptation on a small open model or an embedding fine-tune) with results report.
- Test suite: unit, integration, and RAG-quality evaluation harness.

## 4. Out of Scope

- Production-grade multi-tenant billing/auth (a minimal API-key auth stub is included for demonstration only).
- Large-scale distributed training (fine-tuning demo runs on a small model/subset to prove methodology, not to reach SOTA accuracy).
- Procuring real cloud accounts — Terraform/IaC is written and `plan`-validated structurally; it is not applied against a live AWS account in this workspace.

## 5. Success Criteria

| # | Criterion | Measure |
|---|---|---|
| 1 | ETL ingests sample corpus without manual intervention | `make ingest` exits 0, documents land in vector store |
| 2 | Hybrid retrieval outperforms single-strategy retrieval | Hit-rate@5 and MRR improve vs. pure-semantic and pure-keyword baselines (see `11_rag_evaluation_report.md`) |
| 3 | Agent completes a multi-step research query | End-to-end trace shows plan → tool calls → synthesized answer |
| 4 | API is documented and testable | OpenAPI spec + passing integration tests |
| 5 | Deployable via containers/K8s with CI/CD | `docker build` succeeds; k8s manifests validate; CI workflow lints, tests, and builds on push |
| 6 | Observability in place | Structured logs + Prometheus metrics endpoint exposed |
| 7 | Fine-tuning methodology demonstrated | Experiment report with before/after metrics |

## 6. Stakeholders / Personas

See `14_personas_raci.md` for the full roster and RACI. Primary hands-on role: **GenAI Developer** (this engagement's target position).

## 7. Timeline

6 months, organized into twelve 2-week sprints — see `07_project_plan_sprints.md`.
