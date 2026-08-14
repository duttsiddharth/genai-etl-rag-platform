# Risk Register

**Author (persona):** Project Manager (compiled with Solution Architect & GenAI Developer input)

| ID | Risk | Category | Likelihood | Impact | Mitigation | Owner |
|---|---|---|---|---|---|---|
| R-1 | LLM hallucination produces plausible-but-wrong answers | Quality | Medium | High | RAG grounding with mandatory citations; groundedness spot-checks in eval harness; agent max-step + "insufficient evidence" fallback | GenAI Developer |
| R-2 | Sensitive/PII data ingested into vector store | Compliance | Medium | High | PII-scrub hook in `transform.py`; access-controlled S3 prefix; audit logging | Solution Architect |
| R-3 | Vendor lock-in to a single cloud's LLM/embedding API | Architecture | Medium | Medium | Provider-agnostic interface in `src/cloud/` and `src/genai/embeddings.py`; config-driven provider swap | Solution Architect |
| R-4 | Latency/cost blow-up from uncontrolled agent loops | Cost/Perf | Medium | Medium | Hard `max_steps` cap, per-run token budget, metrics alert on `agent_steps_used` | MLOps Engineer |
| R-5 | Context window overflow on large documents | Technical | Medium | Medium | Chunking with overlap; retrieval top-k capped; summarization fallback for oversized contexts | GenAI Developer |
| R-6 | Vector store index grows stale as documents change | Data quality | Medium | Medium | Manifest checksums detect changed docs; incremental re-ingest on checksum mismatch | GenAI Developer |
| R-7 | Fine-tuned model overfits small demo dataset | ML | High (for the demo scope) | Low (demo is explicitly scoped as a methodology proof) | Documented in `10_finetuning_experiment_report.md` as a known limitation with a path to scale | GenAI Developer |
| R-8 | Kubernetes rollout causes downtime | Ops | Low | High | `RollingUpdate` with `maxUnavailable: 0`, readiness probes, rollback runbook | MLOps Engineer |
| R-9 | Credential leakage via committed secrets | Security | Low | High | `.env.example` only, `.gitignore` excludes `.env`, secrets sourced from cloud secret manager in deployed environments | MLOps Engineer |
| R-10 | Scope creep beyond 6-month contract window | Delivery | Medium | Medium | Sprint-based backlog with fixed Definition of Done; charter explicitly lists out-of-scope items | Product Owner |

## Risk Heat Summary

- **High-impact, needs continuous mitigation:** R-1, R-2, R-8, R-9
- **Medium-impact, monitored via metrics/alerts:** R-3, R-4, R-5, R-6, R-10
- **Accepted for demo scope:** R-7
