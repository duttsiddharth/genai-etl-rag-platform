"""LLM provider abstraction for generation (RAG answers + agent planning).

Persona: GenAI Developer.
JD requirement covered: "Develop, implement, and maintain APIs to
integrate GenAI models into applications and workflows."

`StubLLMProvider` is a deterministic, offline, extractive generator used
by default so this repository is fully runnable without any API key or
network access — a common and legitimate engineering pattern for local
dev/test/CI environments. Cloud/managed chat providers (OpenAI, Bedrock
Claude, Azure OpenAI, Vertex AI Gemini) implement the same interface and
activate the moment credentials are configured via `LLM_PROVIDER`.
"""
from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod

logger = logging.getLogger("documind.genai.llm")

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, prompt: str, *, max_tokens: int = 512) -> str:
        ...


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


class StubLLMProvider(LLMProvider):
    """Deterministic, dependency-free, offline generator.

    Strategy: given a grounded prompt containing CONTEXT and QUESTION
    sections, it performs extractive summarization — ranking context
    sentences by lexical overlap with the question and returning the
    top-ranked sentences as the answer. This keeps the reference
    implementation honest (it can only answer from the provided context,
    never hallucinate beyond it) and fully offline-runnable.
    """

    name = "stub"

    def generate(self, prompt: str, *, max_tokens: int = 512) -> str:
        context_match = re.search(r"CONTEXT:\n(.*?)\nQUESTION:", prompt, re.DOTALL)
        question_match = re.search(r"QUESTION:\n(.*?)(\nANSWER:|$)", prompt, re.DOTALL)

        if not context_match or not question_match:
            # Not a RAG-style prompt (e.g. agent planning) — handled by callers
            # via structured prompts; fall back to echoing a bounded excerpt.
            return prompt.strip()[:max_tokens]

        context = context_match.group(1).strip()
        question = question_match.group(1).strip()
        question_tokens = _tokenize(question)

        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(context) if s.strip()]
        if not sentences:
            return "I don't have enough information in the retrieved context to answer that."

        scored = []
        for s in sentences:
            overlap = len(_tokenize(s) & question_tokens)
            scored.append((overlap, s))
        scored.sort(key=lambda x: -x[0])

        top = [s for score, s in scored if score > 0][:3]
        if not top:
            return "I don't have enough information in the retrieved context to answer that."
        answer = " ".join(top)
        return answer[:max_tokens]


class OpenAIChatProvider(LLMProvider):
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    def generate(self, prompt: str, *, max_tokens: int = 512) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package not installed. `pip install openai`.") from exc
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set.")
        client = OpenAI()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""


class BedrockClaudeProvider(LLMProvider):
    name = "bedrock"

    def __init__(self, model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0", region: str | None = None):
        self.model_id = model_id
        self.region = region or os.getenv("AWS_REGION", "us-east-1")

    def generate(self, prompt: str, *, max_tokens: int = 512) -> str:
        import json as _json

        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("boto3 not installed. `pip install boto3`.") from exc

        client = boto3.client("bedrock-runtime", region_name=self.region)
        body = _json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
        )
        response = client.invoke_model(modelId=self.model_id, body=body)
        payload = _json.loads(response["body"].read())
        return payload["content"][0]["text"]


class AzureOpenAIChatProvider(LLMProvider):
    name = "azure-openai"

    def __init__(self, deployment: str | None = None):
        self.deployment = deployment or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")

    def generate(self, prompt: str, *, max_tokens: int = 512) -> str:
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
        resp = client.chat.completions.create(
            model=self.deployment,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""


class VertexAIChatProvider(LLMProvider):
    name = "vertex-ai"

    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.model_name = model_name

    def generate(self, prompt: str, *, max_tokens: int = 512) -> str:
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
        except ImportError as exc:
            raise RuntimeError("google-cloud-aiplatform not installed. `pip install google-cloud-aiplatform`.") from exc
        project = os.getenv("GCP_PROJECT_ID")
        if not project:
            raise RuntimeError("GCP_PROJECT_ID must be set.")
        vertexai.init(project=project, location=os.getenv("GCP_REGION", "us-central1"))
        model = GenerativeModel(self.model_name)
        response = model.generate_content(prompt)
        return response.text


_REGISTRY: dict[str, type[LLMProvider]] = {
    "stub": StubLLMProvider,
    "openai": OpenAIChatProvider,
    "bedrock": BedrockClaudeProvider,
    "azure-openai": AzureOpenAIChatProvider,
    "vertex-ai": VertexAIChatProvider,
}


def get_llm_provider(name: str | None = None) -> LLMProvider:
    provider_name = name or os.getenv("LLM_PROVIDER", "stub")
    if provider_name not in _REGISTRY:
        raise ValueError(f"Unknown LLM_PROVIDER '{provider_name}'. Options: {list(_REGISTRY)}")
    logger.info("llm.provider_selected", extra={"provider": provider_name})
    return _REGISTRY[provider_name]()
