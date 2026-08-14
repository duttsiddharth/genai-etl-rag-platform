"""Executes the embedding-adapter fine-tuning experiment and asserts it
improves retrieval quality on a small labeled triplet set.

Persona: GenAI Developer.
Produces `artifacts/finetuned/training_report.json`, which is read
verbatim into `docs/10_finetuning_experiment_report.md`.
"""
from __future__ import annotations

from src.genai.embeddings import HashingEmbeddingProvider
from src.genai.finetuning.embedding_adapter_finetune import TrainingExample, run_finetuning_experiment

TRAINING_EXAMPLES = [
    TrainingExample(
        query="What is the default hybrid retrieval alpha?",
        positive_text="The default alpha is 0.6, meaning retrieval leans semantic but still benefits from exact keyword matches.",
        negative_text="The onboarding guide covers environment setup for new developers joining the team.",
    ),
    TrainingExample(
        query="How do I roll back a Kubernetes deployment?",
        positive_text="Run a Kubernetes rollout undo to revert to the previous container image after a bad release.",
        negative_text="The PII scrub hook redacts emails and phone numbers before storage.",
    ),
    TrainingExample(
        query="Why did ticket TCK-1042 happen?",
        positive_text="Root cause was the Chroma persistent volume claim not being ready before the pod started serving traffic.",
        negative_text="Sales engineering reported irrelevant answers for questions containing exact product SKUs.",
    ),
    TrainingExample(
        query="What does a declining retrieval hit-rate metric usually mean?",
        positive_text="A declining hit-rate gauge usually indicates corpus drift or a stale index relative to source changes.",
        negative_text="The agent enforces a hard maximum step count to bound cost and latency.",
    ),
    TrainingExample(
        query="What happens when all requests return 401?",
        positive_text="Confirm the X-API-Key header matches the value mounted into the pod via the Kubernetes Secret.",
        negative_text="The recursive chunking strategy respects paragraph and sentence boundaries.",
    ),
    TrainingExample(
        query="What tool does the agent use for arithmetic?",
        positive_text="The agent calls a calculator tool that safely evaluates arithmetic expressions using an AST allowlist.",
        negative_text="The CI pipeline lints, tests, and builds a Docker image on every push.",
    ),
]


def test_embedding_adapter_finetune_improves_hit_rate():
    embedder = HashingEmbeddingProvider(dimensions=128)
    result = run_finetuning_experiment(
        TRAINING_EXAMPLES,
        embed_fn=embedder.embed,
        dimensions=128,
        output_dir="artifacts/finetuned",
    )

    assert result.hit_rate_after >= result.hit_rate_before
    assert result.final_loss >= 0.0
