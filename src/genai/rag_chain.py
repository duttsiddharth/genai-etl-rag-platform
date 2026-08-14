"""RAG chain: retrieve -> build grounded prompt -> generate -> cite.

Persona: GenAI Developer.
JD requirement covered: "RAG implementation."
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from src.genai.hybrid_retriever import HybridRetriever
from src.genai.llm import LLMProvider

logger = logging.getLogger("documind.genai.rag_chain")

PROMPT_TEMPLATE = """You are a helpful assistant. Answer the QUESTION using ONLY the CONTEXT below.
If the answer is not contained in the CONTEXT, say you don't have enough information.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


@dataclass
class RAGAnswer:
    answer: str
    citations: list[dict] = field(default_factory=list)
    retrieval_strategy: str = "hybrid"
    latency_ms: float = 0.0


class RAGChain:
    def __init__(self, retriever: HybridRetriever, llm: LLMProvider):
        self.retriever = retriever
        self.llm = llm

    def answer(self, question: str, k: int = 4, alpha: float | None = None) -> RAGAnswer:
        start = time.time()
        results = self.retriever.retrieve(question, k=k, alpha=alpha)

        if not results:
            return RAGAnswer(
                answer="I don't have enough information in the retrieved context to answer that.",
                citations=[],
                latency_ms=round((time.time() - start) * 1000, 1),
            )

        context = "\n\n".join(f"[{r.chunk.chunk_id}] {r.chunk.text}" for r in results)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        raw_answer = self.llm.generate(prompt)

        citations = [
            {
                "chunk_id": r.chunk.chunk_id,
                "score": round(r.fused_score, 4),
                "source": r.chunk.metadata.get("source", ""),
            }
            for r in results
        ]

        latency_ms = round((time.time() - start) * 1000, 1)
        logger.info(
            "rag_chain.answer",
            extra={"question": question, "k": k, "citations": len(citations), "latency_ms": latency_ms},
        )
        return RAGAnswer(answer=raw_answer, citations=citations, latency_ms=latency_ms)
