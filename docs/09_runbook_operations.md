# Operations Runbook

**Author (persona):** MLOps/DevOps Engineer & Technical Writer

## 1. Deploy

```bash
# Build & push image (CI does this automatically on main)
docker build -t documind-api:local -f infra/docker/Dockerfile .

# Local run
docker compose -f infra/docker/docker-compose.yml up

# Kubernetes
kubectl apply -f infra/k8s/configmap.yaml
kubectl apply -f infra/k8s/deployment.yaml
kubectl apply -f infra/k8s/service.yaml
kubectl apply -f infra/k8s/hpa.yaml
kubectl rollout status deployment/documind-api
```

## 2. Rollback

```bash
kubectl rollout undo deployment/documind-api
kubectl rollout status deployment/documind-api
```

If the vector store index itself is bad (e.g., a bad ingest), restore from the last-known-good S3 manifest snapshot (S3 versioning enabled by the Terraform module) and re-run `/ingest` on the affected source paths only.

## 3. Common Incidents

| Symptom | Likely Cause | Resolution |
|---|---|---|
| `/health` returns 503 | Vector store unreachable | Check pod logs (`kubectl logs deploy/documind-api`); verify `VECTOR_STORE` config and mounted volume/connection string |
| `/query` latency spikes (P95 > 3s) | Cold embedding model load, or upstream LLM provider throttling | Check `request_latency_seconds` histogram in `/metrics`; check LLM provider status; confirm HPA scaled up |
| `retrieval_hit_rate` gauge trending down | Corpus drift, stale index, or embedding model mismatch | Re-run evaluation harness (`tests/evaluation`), consider re-ingest or re-tune |
| Agent run hits `max_steps` frequently | Goal too complex for available tools, or planner prompt regression | Inspect `steps[]` trace in the response; review recent prompt template changes |
| 401 on all requests | Missing/incorrect `X-API-Key` | Confirm `Secret` mounted correctly in the pod env |
| CI build fails at image push | Registry credentials expired | Rotate registry credentials in GitHub Environment secrets |

## 4. On-call Escalation (design)

1. Alert fires (see `docs/06_mlops_devops_plan.md` §6) → on-call engineer acknowledges.
2. Check `/health` and `/metrics`, then pod logs.
3. If data-related, follow §3 rollback for ingestion.
4. If code-related, follow §2 rollback.
5. File a postmortem using the P0-P3 severity scale in `05_test_plan.md`.

## 5. Routine Maintenance

- Weekly: re-run `tests/evaluation` harness against production-shaped sample queries; compare against the recorded baseline in `docs/11_rag_evaluation_report.md`.
- Monthly: review IAM role permissions for least-privilege drift; rotate API keys.
- Per release: confirm `CHANGELOG`/manifest metadata versions bumped (embedding model, prompt template).
