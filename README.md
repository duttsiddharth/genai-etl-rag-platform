# DocuMind AI — Enterprise Hybrid-RAG & Agentic Knowledge Platform

A reference GenAI Developer project built end-to-end to satisfy every requirement in a target job description (see `docs/13_jd_traceability_matrix.md`): Python ETL for GenAI applications, cloud platform integration (AWS/Azure/GCP), APIs that integrate GenAI models into workflows, hybrid RAG optimization, AI agent orchestration, document storage/chunking/vector databases, basic fine-tuning methods, and DevOps/MLOps practices (Docker, Kubernetes, CI/CD, monitoring/logging).

This is a **portfolio-grade, fully runnable reference implementation** — not slideware. Every architectural claim in `docs/` is backed by code in `src/`, exercised by tests in `tests/`, and verified in `EVIDENCE.md` (a captured record of an actual end-to-end run).

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# End-to-end demo: ingest sample docs -> hybrid RAG query -> agent run
python scripts/run_pipeline_demo.py

# Start the API
uvicorn src.api.main:app --reload
# then open http://localhost:8000/docs

# Run the test suite (unit + integration + evaluation harness)
pytest -v --cov=src
```

Everything above runs **fully offline with zero external API keys** — default providers are a dependency-free hashing embedding model, a numpy-backed vector store, and a deterministic extractive "stub" LLM. Swapping in a real cloud provider (OpenAI, AWS Bedrock, Azure OpenAI, GCP Vertex AI) is a one-line environment variable change — see `.env.example`.

## Project Artifacts (start to end)

| Phase | Artifacts |
|---|---|
| Initiation | [`docs/01_project_charter.md`](docs/01_project_charter.md) |
| Requirements | [`docs/02_requirements_brd_frd.md`](docs/02_requirements_brd_frd.md) |
| Design | [`docs/03_architecture_design.md`](docs/03_architecture_design.md), [`docs/04_api_specification.md`](docs/04_api_specification.md) |
| Planning | [`docs/07_project_plan_sprints.md`](docs/07_project_plan_sprints.md), [`docs/08_risk_register.md`](docs/08_risk_register.md), [`docs/14_personas_raci.md`](docs/14_personas_raci.md) |
| Build | [`src/`](src/) — ETL, GenAI core, API, cloud integrations |
| Quality | [`docs/05_test_plan.md`](docs/05_test_plan.md), [`tests/`](tests/), [`docs/11_rag_evaluation_report.md`](docs/11_rag_evaluation_report.md) |
| R&D | [`docs/10_finetuning_experiment_report.md`](docs/10_finetuning_experiment_report.md) |
| Release Engineering | [`infra/`](infra/), [`docs/06_mlops_devops_plan.md`](docs/06_mlops_devops_plan.md) |
| Operate | [`docs/09_runbook_operations.md`](docs/09_runbook_operations.md) |
| Demonstrate | [`docs/12_demo_script.md`](docs/12_demo_script.md), [`EVIDENCE.md`](EVIDENCE.md) |
| **JD Traceability** | [`docs/13_jd_traceability_matrix.md`](docs/13_jd_traceability_matrix.md) — every JD line item mapped to the file that proves it |

## Repository Structure

```
docs/            Persona-authored artifacts: charter, requirements, architecture, test plan,
                 MLOps plan, sprint plan, risk register, runbook, experiment reports, demo script
src/etl/         Extract / transform / load pipeline (PDF, HTML, TXT, JSON sources)
src/genai/       Chunking, embeddings, vector store, hybrid retriever, RAG chain,
                 agent orchestrator + tools, fine-tuning experiments
src/api/         FastAPI service: /ingest, /query, /agent/run, /health, /metrics
src/cloud/       Provider-agnostic cloud storage interface + AWS/Azure/GCP implementations
src/monitoring/  Structured JSON logging + Prometheus metrics
infra/docker/    Dockerfile + docker-compose for local/containerized runs
infra/k8s/       Kubernetes Deployment/Service/HPA/ConfigMap manifests
infra/terraform/ AWS infrastructure as code (S3, ECR, IAM, CloudWatch)
infra/ci-cd/     GitHub Actions CI/CD pipeline (mirrored at .github/workflows/ci-cd.yaml,
                 which is the copy GitHub Actions actually triggers on)
tests/           Unit, integration, and RAG/fine-tuning evaluation harnesses
data/sample_docs/ Sample heterogeneous corpus used by the demo and evaluation harness
scripts/         `run_pipeline_demo.py` — end-to-end runnable demo
```

## Personas

This project is documented as if delivered by a cross-functional team (Product Owner, Business Analyst, Solution Architect, GenAI Developer, MLOps/DevOps Engineer, QA Engineer, Project Manager, Technical Writer) — see [`docs/14_personas_raci.md`](docs/14_personas_raci.md) for who owns what. The GenAI Developer persona — the target role — is the primary hands-on contributor across `src/etl/`, `src/genai/`, `src/api/`, and the fine-tuning experiment.
