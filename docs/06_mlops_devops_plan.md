# DevOps / MLOps Plan

**Author (persona):** MLOps / DevOps Engineer

## 1. Branching & CI/CD Strategy

- Trunk-based development: short-lived feature branches → PR → `main`.
- CI (`.github/workflows/ci-cd.yaml`) runs on every push/PR:
  1. Install deps (cached).
  2. Lint (`ruff`).
  3. Unit + integration tests (`pytest`, with coverage).
  4. Build Docker image.
  5. (On `main`) Push image to registry (ECR/ACR/Artifact Registry — placeholder credentials) and tag with commit SHA + `latest`.
- CD: image promoted to a Kubernetes rolling deployment; manual approval gate for production namespace (represented as a GitHub Environment protection rule).

## 2. Containerization Strategy

- Multi-stage `Dockerfile`: builder stage installs deps, runtime stage copies only the app + venv (smaller, no build toolchain in the final image).
- Non-root user, `HEALTHCHECK` against `/health`.
- `docker-compose.yml` for local dev: API service + a placeholder for a persistent vector store volume.

## 3. Kubernetes Deployment Strategy

- `Deployment` with `RollingUpdate` (`maxSurge: 1`, `maxUnavailable: 0`) for zero-downtime releases.
- `readinessProbe`/`livenessProbe` on `/health`.
- `HorizontalPodAutoscaler` targets 70% CPU, min 2 / max 6 replicas.
- Config via `ConfigMap` (non-secret) + `Secret` (API keys, cloud credentials) — never baked into the image.
- Rollback: `kubectl rollout undo deployment/documind-api` documented in the runbook; CD pipeline keeps the last 5 ReplicaSets.

## 4. Infrastructure as Code

- Terraform (`infra/terraform/`) provisions, for AWS:
  - S3 bucket for raw documents + ingestion manifests (versioned, encrypted).
  - ECR repository for the container image.
  - IAM role with least-privilege policy (S3 read/write on the project prefix, Bedrock `InvokeModel` only).
  - CloudWatch log group with retention policy.
- Variables parameterize region/environment so the same module deploys dev/stage/prod.
- State is designed for a remote backend (S3 + DynamoDB lock) — declared as a `backend` block placeholder, not applied in this sandbox.

## 5. Model / Pipeline Lifecycle Management

| Stage | Practice |
|---|---|
| Versioning | Embedding model name + version, vector store schema version, and prompt template version are all recorded in `manifest` metadata written during ingestion |
| Registry (conceptual) | Fine-tuned adapters (LoRA weights) are saved with a name+timestamp under `artifacts/finetuned/`, analogous to registering a model version |
| Promotion gates | CI must pass unit+integration tests; evaluation harness (`tests/evaluation`) must not regress hit-rate@5 by more than 5% vs. the last recorded baseline before a fine-tuned/embedding change is "promoted" |
| Drift monitoring | `src/monitoring/metrics.py` exposes a `retrieval_hit_rate` gauge and `low_confidence_answers_total` counter; a sustained drop / rise is the trigger to re-run evaluation and consider re-tuning |
| Rollback | Previous container image tag + previous vector-store snapshot (S3 versioning) allow reverting a bad release |

## 6. Monitoring & Logging

- **Logging:** structured JSON via `src/monitoring/logger.py`, one line per pipeline stage and per API request, correlation id propagated through ETL → retrieval → generation.
- **Metrics:** `src/monitoring/metrics.py` exposes Prometheus counters/histograms: `http_requests_total`, `request_latency_seconds`, `retrieval_hit_rate`, `tokens_used_total`, `agent_steps_used`.
- **Dashboards (design):** Grafana dashboard JSON stub (`infra/monitoring/grafana-dashboard.json`) with panels for latency P50/P95, error rate, retrieval hit-rate trend, token spend.
- **Alerting (design):** Alert rules sketch in `infra/monitoring/alerts.md` — error rate > 5% for 5m, P95 latency > 3s for 10m, hit-rate gauge < 0.5 for 30m.

## 7. Operational Cadence

- Sprint-end demo + retro (see `07_project_plan_sprints.md`).
- Weekly evaluation-harness re-run against the growing corpus to catch retrieval regressions early.
