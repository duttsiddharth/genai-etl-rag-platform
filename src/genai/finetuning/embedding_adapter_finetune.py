"""Basic fine-tuning method #1 (executed in this repo): a lightweight,
parameter-efficient linear adapter trained on top of frozen embeddings.

Persona: GenAI Developer.
JD requirement covered: "Apply expertise in ... basic finetuning methods."

Why an adapter instead of full model fine-tuning: this mirrors the
*parameter-efficient fine-tuning* (PEFT) philosophy used in production
(the same idea behind LoRA) — freeze the large pretrained
representation, train a small number of additional parameters on top of
it. Here the "large pretrained representation" is the embedding
provider's output vector, and the adapter is a single learned linear
projection matrix W (dimensions x dimensions), trained with a
contrastive objective (pull query embeddings toward their labeled
positive chunk, push away from a labeled negative chunk).

This is intentionally implemented in pure NumPy (no torch/transformers)
so it trains in well under a second on CPU with zero extra downloads —
appropriate for a "basic finetuning methods" demonstration. The
transformer-level LoRA reference implementation is provided alongside
this file in `lora_finetune_demo.py` for environments with GPU + the
full HF/PEFT stack available.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger("documind.genai.finetuning.embedding_adapter")


@dataclass
class TrainingExample:
    query: str
    positive_text: str
    negative_text: str


@dataclass
class AdapterTrainingResult:
    epochs: int
    final_loss: float
    weight_path: str
    hit_rate_before: float
    hit_rate_after: float


class LinearAdapter:
    """y = normalize(W @ x). Initialized near-identity so training starts
    from the frozen embedding's behavior and only nudges it."""

    def __init__(self, dimensions: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.W = np.eye(dimensions) + rng.normal(0, 0.01, size=(dimensions, dimensions))

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        out = vectors @ self.W.T
        norms = np.linalg.norm(out, axis=-1, keepdims=True)
        norms[norms == 0] = 1e-8
        return out / norms

    def save(self, path: str | Path) -> None:
        np.save(str(path), self.W)

    @classmethod
    def load(cls, path: str | Path, dimensions: int) -> "LinearAdapter":
        adapter = cls(dimensions)
        adapter.W = np.load(str(path))
        return adapter


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-8
    return float(np.dot(a, b) / denom)


def train_adapter(
    examples: list[TrainingExample],
    embed_fn,
    dimensions: int,
    epochs: int = 200,
    lr: float = 0.05,
    margin: float = 0.2,
    seed: int = 42,
) -> tuple[LinearAdapter, list[float]]:
    """Train a linear adapter with a numeric-gradient contrastive triplet loss:
        loss = max(0, margin - cos(Wq, Wpos) + cos(Wq, Wneg))
    Uses simple finite-difference-free analytic gradient of the cosine terms
    w.r.t. W for a compact, dependency-free training loop.
    """
    adapter = LinearAdapter(dimensions, seed=seed)

    queries = np.array(embed_fn([e.query for e in examples]))
    positives = np.array(embed_fn([e.positive_text for e in examples]))
    negatives = np.array(embed_fn([e.negative_text for e in examples]))

    loss_history: list[float] = []
    W = adapter.W.copy()

    for epoch in range(epochs):
        grad = np.zeros_like(W)
        total_loss = 0.0

        for q, p, n in zip(queries, positives, negatives):
            wq = W @ q
            wp = W @ p
            wn = W @ n

            wq_n, wp_n, wn_n = (v / (np.linalg.norm(v) or 1e-8) for v in (wq, wp, wn))
            sim_pos = float(np.dot(wq_n, wp_n))
            sim_neg = float(np.dot(wq_n, wn_n))
            loss = max(0.0, margin - sim_pos + sim_neg)
            total_loss += loss

            if loss > 0:
                # Approximate gradient: push W q closer to p's direction,
                # further from n's direction (outer-product update).
                grad += lr * (np.outer(p, q) - np.outer(n, q)) / len(examples)

        W = W + grad
        avg_loss = total_loss / len(examples)
        loss_history.append(avg_loss)
        if epoch % 40 == 0 or epoch == epochs - 1:
            logger.info("finetune.epoch", extra={"epoch": epoch, "avg_loss": round(avg_loss, 5)})

    adapter.W = W
    return adapter, loss_history


def evaluate_hit_rate(
    examples: list[TrainingExample],
    embed_fn,
    adapter: LinearAdapter | None,
) -> float:
    """For each example, hit = 1 if positive is closer to query than negative."""
    hits = 0
    for ex in examples:
        q, p, n = embed_fn([ex.query, ex.positive_text, ex.negative_text])
        q, p, n = np.array(q), np.array(p), np.array(n)
        if adapter is not None:
            q, p, n = adapter.transform(np.array([q, p, n]))
        sim_pos = _cosine(q, p)
        sim_neg = _cosine(q, n)
        if sim_pos > sim_neg:
            hits += 1
    return hits / len(examples) if examples else 0.0


def run_finetuning_experiment(
    examples: list[TrainingExample],
    embed_fn,
    dimensions: int,
    output_dir: str | Path = "artifacts/finetuned",
) -> AdapterTrainingResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hit_rate_before = evaluate_hit_rate(examples, embed_fn, adapter=None)
    adapter, loss_history = train_adapter(examples, embed_fn, dimensions)
    hit_rate_after = evaluate_hit_rate(examples, embed_fn, adapter=adapter)

    weight_path = output_dir / "embedding_adapter.npy"
    adapter.save(weight_path)

    result = AdapterTrainingResult(
        epochs=len(loss_history),
        final_loss=loss_history[-1] if loss_history else float("nan"),
        weight_path=str(weight_path),
        hit_rate_before=hit_rate_before,
        hit_rate_after=hit_rate_after,
    )

    report_path = output_dir / "training_report.json"
    report_path.write_text(json.dumps(result.__dict__, indent=2))
    logger.info("finetune.experiment_complete", extra=result.__dict__)
    return result
