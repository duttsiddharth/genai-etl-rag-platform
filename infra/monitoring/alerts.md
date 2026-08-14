# Alert Rules (design)

Persona: MLOps/DevOps Engineer. Referenced from `docs/06_mlops_devops_plan.md` §6.

| Alert | Condition | Severity | Action |
|---|---|---|---|
| High error rate | `rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05` for 5m | P1 | Page on-call, check `09_runbook_operations.md` §3 |
| High latency | `histogram_quantile(0.95, request_latency_seconds_bucket) > 3` for 10m | P2 | Check embedding/LLM provider status, check HPA scale-out |
| Retrieval quality drop | `retrieval_hit_rate < 0.5` for 30m | P2 | Re-run evaluation harness, consider re-ingest/re-tune |
| Elevated low-confidence answers | `rate(low_confidence_answers_total[15m]) > 0.2` | P3 | Review recent queries for corpus gaps |
| Agent step exhaustion | `histogram_quantile(0.9, agent_steps_used_bucket) >= max_steps` | P3 | Review planner heuristic / prompt regression |
