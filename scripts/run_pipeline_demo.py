#!/usr/bin/env python3
"""End-to-end demo runner: ETL ingest -> hybrid RAG query -> agent run.

Persona: GenAI Developer.
Used both for local demoing (see docs/12_demo_script.md) and to generate
the EVIDENCE.md artifact captured by the verification step of this
project's build.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.etl.pipeline import ETLPipeline
from src.genai.agents.orchestrator import AgentOrchestrator
from src.genai.agents.tools import CalculatorTool, RetrievalTool, WebLookupTool
from src.genai.embeddings import get_embedding_provider
from src.genai.hybrid_retriever import HybridRetriever
from src.genai.llm import get_llm_provider
from src.genai.rag_chain import RAGChain
from src.genai.vector_store import get_vector_store
from src.monitoring.logger import configure_logging

configure_logging()


def main() -> None:
    print("=" * 80)
    print("DocuMind AI — End-to-End Pipeline Demo")
    print("=" * 80)

    embedding_provider = get_embedding_provider()
    vector_store = get_vector_store()
    llm = get_llm_provider()

    print(f"\n[config] embedding_provider={embedding_provider.name} vector_store={vector_store.name} llm={llm.name}\n")

    pipeline = ETLPipeline(embedding_provider=embedding_provider, vector_store=vector_store)
    t0 = time.time()
    summary = pipeline.run_directory("data/sample_docs", force=True)
    print(f"[ingest] documents_processed={summary.documents_processed} chunks_indexed={summary.chunks_indexed} "
          f"skipped_unchanged={summary.skipped_unchanged} duration={summary.duration_seconds}s")

    retriever = HybridRetriever(vector_store, embedding_provider, alpha=0.6)
    rag_chain = RAGChain(retriever, llm)

    demo_questions = [
        "What is the default hybrid retrieval alpha and what does it control?",
        "What is the rollback procedure if a release causes elevated error rates?",
        "What caused ticket TCK-1042 and how was it resolved?",
    ]

    print("\n" + "-" * 80)
    print("Hybrid RAG query demo")
    print("-" * 80)
    for q in demo_questions:
        result = rag_chain.answer(q, k=4)
        print(f"\nQ: {q}")
        print(f"A: {result.answer}")
        print(f"   citations: {[c['chunk_id'] for c in result.citations]}")
        print(f"   latency_ms: {result.latency_ms}")

    print("\n" + "-" * 80)
    print("Agent orchestrator demo")
    print("-" * 80)
    tools = [RetrievalTool(retriever), CalculatorTool(), WebLookupTool()]
    agent = AgentOrchestrator(tools, llm, max_steps=6)

    goal = "Compare the rollback procedure and the vector store connection timeout incident, then summarize the operational lesson."
    agent_result = agent.run(goal)
    print(f"\nGoal: {goal}")
    for step in agent_result.steps:
        obs_preview = (step.observation[:120] + "...") if step.observation and len(step.observation) > 120 else step.observation
        print(f"  step {step.step}: action={step.action} args={step.args} observation={obs_preview}")
    print(f"\nFinal answer: {agent_result.answer}")
    print(f"steps_used={agent_result.steps_used}/{agent_result.max_steps} latency_ms={agent_result.latency_ms}")

    print(f"\n[total demo duration] {round(time.time() - t0, 2)}s")


if __name__ == "__main__":
    main()
