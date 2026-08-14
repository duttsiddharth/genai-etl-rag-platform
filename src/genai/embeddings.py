"""Embedding provider abstraction.

Persona: GenAI Developer.
JD requirements covered: "Collaborate with cloud platforms (AWS, Azure,
GCP) to build GenAI applications" and "vector databases / RAG
implementation."

Design: a single `EmbeddingProvider` interface with one required method,
`embed(texts) -> list[vector]`. `get_embedding_provider()` selects an
implementation from the `EMBEDDING_PROVIDER` env var so application code
never imports a specific vendor SDK directly.

Default provider (`local-hashing`) is dependency-free and fully
deterministic, so the reference implementation in this repository runs
end-to-end with no external API keys and no multi-gigabyte model
downloads. `local-sentence-transformers` is also provided for
environments where that optional dependency is installed. The cloud
providers are complete, correct implementations that activate the moment
credentials are supplied — swapping providers is a one-line config
change, not a code change (see NFR-6, "Portability").
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from abc import ABC, abstractmethod
from collections import Counter

logger = logging.getLogger("documind.genai.embeddings")

DEFAULT_DIM = 384
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class EmbeddingProvider(ABC):
    name: str = "base"
    dimensions: int = DEFAULT_DIM

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class HashingEmbeddingProvider(EmbeddingProvider):
    """A deterministic, dependency-free embedding provider.

    Implements a classic feature-hashing (a.k.a. "hashing trick") bag-of-
    words vectorizer with sublinear TF weighting and L2 normalization —
    the same family of technique used by `sklearn.HashingVectorizer` and
    common as a lightweight/offline fallback in production RAG systems
    before falling back further to a full neural embedding model.

    This is intentionally NOT a neural embedding model; it is documented
    here as the "local, zero-dependency, zero-cost" tier of the provider
    ladder, with `LocalSentenceTransformerEmbeddingProvider` as the
    drop-in neural upgrade when `sentence-transformers` is installed.
    """

    name = "local-hashing"

    def __init__(self, dimensions: int = DEFAULT_DIM):
        self.dimensions = dimensions

    def _vectorize_one(self, text: str) -> list[float]:
        import numpy as np

        tokens = _tokenize(text)
        vec = np.zeros(self.dimensions, dtype=np.float64)
        if not tokens:
            return vec.tolist()

        counts = Counter(tokens)
        for token, count in counts.items():
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % self.dimensions
            sign = 1.0 if int(digest[8], 16) % 2 == 0 else -1.0
            weight = 1.0 + math.log(count)  # sublinear TF
            vec[idx] += sign * weight

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorize_one(t) for t in texts]


class LocalSentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Neural embeddings via `sentence-transformers` (optional dependency).

    Activated with `EMBEDDING_PROVIDER=local-sentence-transformers`.
    Falls back with a clear error if the package isn't installed, rather
    than silently degrading, since callers may specifically require
    embedding-quality parity with a shipped fine-tuned adapter.
    """

    name = "local-sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install with `pip install sentence-transformers` or set "
                "EMBEDDING_PROVIDER=local-hashing to use the dependency-free provider."
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.dimensions = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Cloud embeddings via the OpenAI API (also compatible with Azure
    OpenAI's embeddings endpoint when `OPENAI_BASE_URL` is set)."""

    name = "openai"

    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self.dimensions = 1536

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package not installed. `pip install openai`.") from exc
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set.")
        client = OpenAI()
        resp = client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]


class BedrockEmbeddingProvider(EmbeddingProvider):
    """Cloud embeddings via AWS Bedrock (Amazon Titan Embeddings)."""

    name = "bedrock"

    def __init__(self, model_id: str = "amazon.titan-embed-text-v2:0", region: str | None = None):
        self.model_id = model_id
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        self.dimensions = 1024

    def embed(self, texts: list[str]) -> list[list[float]]:
        import json as _json

        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("boto3 not installed. `pip install boto3`.") from exc

        client = boto3.client("bedrock-runtime", region_name=self.region)
        vectors = []
        for text in texts:
            body = _json.dumps({"inputText": text})
            response = client.invoke_model(modelId=self.model_id, body=body)
            payload = _json.loads(response["body"].read())
            vectors.append(payload["embedding"])
        return vectors


class AzureOpenAIEmbeddingProvider(EmbeddingProvider):
    """Cloud embeddings via Azure OpenAI Service."""

    name = "azure-openai"

    def __init__(self, deployment: str | None = None):
        self.deployment = deployment or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
        self.dimensions = 1536

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            from openai import AzureOpenAI
        except ImportError as exc:
            raise RuntimeError("openai package not installed. `pip install openai`.") from exc
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not endpoint or not api_key:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set.")
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        )
        resp = client.embeddings.create(model=self.deployment, input=texts)
        return [d.embedding for d in resp.data]


class VertexAIEmbeddingProvider(EmbeddingProvider):
    """Cloud embeddings via Google Cloud Vertex AI."""

    name = "vertex-ai"

    def __init__(self, model_name: str = "text-embedding-004"):
        self.model_name = model_name
        self.dimensions = 768

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            import vertexai
            from vertexai.language_models import TextEmbeddingModel
        except ImportError as exc:
            raise RuntimeError("google-cloud-aiplatform not installed. `pip install google-cloud-aiplatform`.") from exc
        project = os.getenv("GCP_PROJECT_ID")
        if not project:
            raise RuntimeError("GCP_PROJECT_ID must be set.")
        vertexai.init(project=project, location=os.getenv("GCP_REGION", "us-central1"))
        model = TextEmbeddingModel.from_pretrained(self.model_name)
        embeddings = model.get_embeddings(texts)
        return [e.values for e in embeddings]


_REGISTRY: dict[str, type[EmbeddingProvider]] = {
    "local-hashing": HashingEmbeddingProvider,
    "local-sentence-transformers": LocalSentenceTransformerEmbeddingProvider,
    "openai": OpenAIEmbeddingProvider,
    "bedrock": BedrockEmbeddingProvider,
    "azure-openai": AzureOpenAIEmbeddingProvider,
    "vertex-ai": VertexAIEmbeddingProvider,
}


def get_embedding_provider(name: str | None = None) -> EmbeddingProvider:
    provider_name = name or os.getenv("EMBEDDING_PROVIDER", "local-hashing")
    if provider_name not in _REGISTRY:
        raise ValueError(f"Unknown EMBEDDING_PROVIDER '{provider_name}'. Options: {list(_REGISTRY)}")
    logger.info("embeddings.provider_selected", extra={"provider": provider_name})
    return _REGISTRY[provider_name]()
