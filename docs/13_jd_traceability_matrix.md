# Job Description → Project Traceability Matrix

Direct mapping from every line in the target job description to the concrete artifact(s) in this repository that demonstrate it.

## Must Have

| JD Requirement | Demonstrated by |
|---|---|
| Design and develop efficient, maintainable, reusable Python scripts for ETL in GenAI applications | `src/etl/extract.py`, `src/etl/transform.py`, `src/etl/load.py`, `src/etl/pipeline.py` — modular, typed, logged, unit-tested ETL for PDF/HTML/TXT/JSON sources |
| Strong proficiency in Python | Entire `src/` tree: type hints, dataclasses, abstract base classes, async FastAPI, pytest suite, packaging via `pyproject.toml`/`requirements.txt` |
| Collaborate with cloud platforms (AWS, Azure, GCP) to build GenAI applications | `src/cloud/aws_integration.py`, `azure_integration.py`, `gcp_integration.py` behind a common `CloudStorageProvider`/`CloudModelProvider` interface; `infra/terraform/` for AWS deployment |
| Develop, implement, and maintain APIs to integrate GenAI models into applications and workflows | `src/api/main.py` (FastAPI), `src/api/routes/*` (`/ingest`, `/query`, `/agent/run`, `/health`, `/metrics`), `docs/04_api_specification.md` |
| Research and experiment with generative AI techniques: AI agents, hybrid RAG optimization, workflow orchestration | `src/genai/agents/orchestrator.py` (plan→act→observe loop), `src/genai/hybrid_retriever.py` (dense+BM25 fusion with tunable alpha), `docs/11_rag_evaluation_report.md` (experiment results) |
| Apply GenAI concepts: document storage, chunking, vector databases, RAG implementation, basic fine-tuning | `src/genai/chunking.py`, `src/genai/vector_store.py`, `src/genai/rag_chain.py`, `src/genai/finetuning/lora_finetune_demo.py`, `docs/10_finetuning_experiment_report.md` |

## Nice to Have

| JD Requirement | Demonstrated by |
|---|---|
| Champion DevOps/MLOps: CI/CD and AI model monitoring | `.github/workflows/ci-cd.yaml`, `src/monitoring/metrics.py`, `docs/06_mlops_devops_plan.md` |
| Docker, Kubernetes, Git for AI pipelines | `infra/docker/Dockerfile`, `infra/docker/docker-compose.yml`, `infra/k8s/*.yaml`, this repo is a Git repository with conventional commits |
| Monitoring and logging for AI model performance/reliability | `src/monitoring/logger.py` (structured JSON logs), `src/monitoring/metrics.py` (Prometheus counters/histograms for latency, token usage, retrieval hit-rate), `docs/09_runbook_operations.md` |
| Collaboration with software engineering and operations teams | `docs/14_personas_raci.md`, sprint demo cadence in `docs/07_project_plan_sprints.md` |
| DevOps/MLOps methodology, CI/CD, AI model lifecycle management | `docs/06_mlops_devops_plan.md` (branching strategy, model registry approach, promotion gates, rollback plan) |

## Artifact Inventory (start-to-end lifecycle)

1. **Initiation** — `01_project_charter.md`
2. **Requirements** — `02_requirements_brd_frd.md`
3. **Design** — `03_architecture_design.md`, `04_api_specification.md`
4. **Planning** — `07_project_plan_sprints.md`, `08_risk_register.md`, `14_personas_raci.md`
5. **Build** — `src/*` (ETL, GenAI core, API, cloud integrations)
6. **Quality** — `05_test_plan.md`, `tests/*`, `11_rag_evaluation_report.md`
7. **R&D** — `10_finetuning_experiment_report.md`
8. **Release Engineering** — `infra/*`, `06_mlops_devops_plan.md`
9. **Operate** — `09_runbook_operations.md`, `src/monitoring/*`
10. **Demonstrate** — `12_demo_script.md`, `EVIDENCE.md` (generated after end-to-end run)
