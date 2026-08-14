# Project Plan — Sprint Breakdown (6-Month Engagement)

**Author (persona):** Project Manager
Cadence: 2-week sprints × 12 = 6 months, matching the contract term.

| Sprint | Dates (relative) | Goal | Key Deliverables | Owner persona(s) |
|---|---|---|---|---|
| 1 | Wk 1-2 | Discovery & Charter | Project charter, stakeholder map, initial requirements draft | Product Owner, Business Analyst |
| 2 | Wk 3-4 | Requirements & Architecture | BRD/FRD signed off, architecture design + diagrams, API spec draft | Business Analyst, Solution Architect |
| 3 | Wk 5-6 | ETL Foundation | Extract/transform/load modules, source-format support (PDF/HTML/TXT/JSON), unit tests | GenAI Developer |
| 4 | Wk 7-8 | Chunking & Embeddings | Chunking strategies, embedding provider abstraction, local + cloud embedding support | GenAI Developer |
| 5 | Wk 9-10 | Vector Store & Baseline Retrieval | Vector store integration, dense-only retrieval baseline, BM25 lexical index | GenAI Developer |
| 6 | Wk 11-12 | Hybrid RAG | Hybrid retriever (score fusion), RAG chain with citations, first evaluation report | GenAI Developer, QA |
| 7 | Wk 13-14 | Agentic Orchestration | Planner/executor agent loop, tool integrations (retrieval, calculator), step-trace logging | GenAI Developer |
| 8 | Wk 15-16 | API & Cloud Integration | FastAPI service, AWS/Azure/GCP provider modules, auth stub | GenAI Developer |
| 9 | Wk 17-18 | Fine-tuning R&D Spike | LoRA/PEFT domain-adaptation experiment, before/after report | GenAI Developer |
| 10 | Wk 19-20 | DevOps/MLOps | Dockerfile, Kubernetes manifests, Terraform, CI/CD pipeline | MLOps Engineer |
| 11 | Wk 21-22 | Observability & Hardening | Structured logging, Prometheus metrics, load-test design, security review | MLOps Engineer, QA |
| 12 | Wk 23-24 | UAT, Demo & Handover | Full regression pass, demo script + walkthrough, runbook, handover docs | Whole team |

## Milestones

- **M1 (end Sprint 2):** Architecture approved — build can start.
- **M2 (end Sprint 6):** Hybrid RAG MVP demonstrable end-to-end.
- **M3 (end Sprint 8):** API + multi-cloud integration complete.
- **M4 (end Sprint 10):** Deployable via CI/CD to Kubernetes.
- **M5 (end Sprint 12):** Go-live readiness / handover.

## Definition of Done (per sprint)

1. Code merged to `main` via reviewed PR.
2. Unit/integration tests passing in CI.
3. Relevant doc(s) updated.
4. Sprint demo delivered to Product Owner.
5. No open P0/P1 defects.
