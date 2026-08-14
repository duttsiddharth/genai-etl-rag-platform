# Personas & RACI

This project is documented as if delivered by a small cross-functional team. Every artifact in `docs/` and `src/` is labeled with the persona that would own it in a real engagement, so the repository demonstrates the *full range* of collaboration the job description implies ("Collaboration experience with software engineering and operations teams for seamless AI model integration and deployment").

## Personas

| Persona | Responsibility in this project | Primary artifacts owned |
|---|---|---|
| **Product Owner / Sponsor** | Defines business objective, prioritizes backlog, accepts sprint demos | `01_project_charter.md`, `07_project_plan_sprints.md` |
| **Business Analyst** | Elicits and documents requirements, acceptance criteria | `02_requirements_brd_frd.md` |
| **Solution Architect** | System design, cloud topology, security architecture | `03_architecture_design.md` |
| **GenAI Developer** (target role) | Builds ETL, chunking/embeddings/vector store, hybrid RAG, agent orchestration, API integration, fine-tuning spike | `src/etl/*`, `src/genai/*`, `src/api/*`, `10_finetuning_experiment_report.md` |
| **MLOps/DevOps Engineer** | CI/CD, containerization, Kubernetes, IaC, monitoring/logging | `infra/*`, `06_mlops_devops_plan.md`, `src/monitoring/*` |
| **QA / Evaluation Engineer** | Test plan, unit/integration tests, RAG evaluation harness | `05_test_plan.md`, `tests/*`, `11_rag_evaluation_report.md` |
| **Project Manager** | Sprint tracking, risk register, status reporting | `07_project_plan_sprints.md`, `08_risk_register.md` |
| **Technical Writer** | Runbook, demo script, README | `09_runbook_operations.md`, `12_demo_script.md`, root `README.md` |

## RACI Matrix

R = Responsible, A = Accountable, C = Consulted, I = Informed

| Artifact / Deliverable | Product Owner | Business Analyst | Solution Architect | GenAI Developer | MLOps Engineer | QA Engineer | PM |
|---|---|---|---|---|---|---|---|
| Project Charter | A | C | C | I | I | I | R |
| Requirements (BRD/FRD) | C | A/R | C | C | I | C | I |
| Architecture Design | I | C | A/R | C | C | I | I |
| ETL Pipeline | I | I | C | A/R | C | C | I |
| Chunking / Embeddings / Vector Store | I | I | C | A/R | I | C | I |
| Hybrid RAG + Agents | I | C | C | A/R | I | C | I |
| API Layer | I | C | C | A/R | C | C | I |
| Fine-tuning Experiment | I | I | C | A/R | I | C | I |
| Docker/K8s/Terraform | I | I | C | C | A/R | I | I |
| CI/CD Pipeline | I | I | C | C | A/R | C | I |
| Monitoring/Logging | I | I | C | C | A/R | C | I |
| Test Plan & Test Suite | I | C | I | C | C | A/R | I |
| RAG Evaluation Report | I | C | I | R | I | A/R | I |
| Runbook | I | I | C | C | R | I | A |
| Sprint Plan / Status | A | I | I | I | I | I | R |
| Risk Register | C | C | C | C | C | C | A/R |

> In this single-author repository, the GenAI Developer persona is the primary hands-on contributor (matching the target job), and every other persona's artifact was authored to the standard that role would need to hand off to / receive from in a real team.
