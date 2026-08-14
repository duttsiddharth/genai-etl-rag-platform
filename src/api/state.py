"""Application state / dependency wiring.

Persona: GenAI Developer.
Builds all GenAI components once at startup (embedding provider, vector
store, hybrid retriever, LLM, RAG chain, agent orchestrator, ETL
pipeline) so requests reuse warm singletons instead of re-initializing
per call.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from src.etl.pipeline import ETLPipeline
from src.genai.agents.orchestrator import AgentOrchestrator
from src.genai.agents.tools import CalculatorTool, RetrievalTool, WebLookupTool
from src.genai.embeddings import EmbeddingProvider, get_embedding_provider
from src.genai.hybrid_retriever import HybridRetriever
from src.genai.llm import LLMProvider, get_llm_provider
from src.genai.rag_chain import RAGChain
from src.genai.vector_store import VectorStore, get_vector_store


@dataclass
class AppState:
    embedding_provider: EmbeddingProvider
    vector_store: VectorStore
    llm: LLMProvider
    retriever: HybridRetriever
    rag_chain: RAGChain
    agent: AgentOrchestrator
    etl_pipeline: ETLPipeline
    api_key: str
    default_source_dir: str


def build_app_state() -> AppState:
    embedding_provider = get_embedding_provider()
    vector_store = get_vector_store()
    llm = get_llm_provider()

    retriever = HybridRetriever(vector_store, embedding_provider, alpha=float(os.getenv("HYBRID_ALPHA", "0.6")))
    rag_chain = RAGChain(retriever, llm)

    tools = [RetrievalTool(retriever), CalculatorTool(), WebLookupTool()]
    agent = AgentOrchestrator(tools, llm, max_steps=int(os.getenv("AGENT_MAX_STEPS", "6")))

    etl_pipeline = ETLPipeline(embedding_provider=embedding_provider, vector_store=vector_store)

    return AppState(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        llm=llm,
        retriever=retriever,
        rag_chain=rag_chain,
        agent=agent,
        etl_pipeline=etl_pipeline,
        api_key=os.getenv("API_KEY", "demo-local-key"),
        default_source_dir=os.getenv("SOURCE_DIR", "data/sample_docs"),
    )
