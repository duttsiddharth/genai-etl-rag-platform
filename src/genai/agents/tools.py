"""Tools available to the agent orchestrator.

Persona: GenAI Developer.
JD requirement covered: "AI agents ... workflow orchestration."
"""
from __future__ import annotations

import ast
import logging
import operator
from abc import ABC, abstractmethod
from typing import Any

from src.genai.hybrid_retriever import HybridRetriever

logger = logging.getLogger("documind.genai.agents.tools")


class Tool(ABC):
    name: str
    description: str

    @abstractmethod
    def run(self, args: dict[str, Any]) -> str:
        ...


class RetrievalTool(Tool):
    name = "retrieve"
    description = "Retrieve relevant knowledge-base passages for a search query. args: {query: str}"

    def __init__(self, retriever: HybridRetriever, k: int = 3):
        self.retriever = retriever
        self.k = k

    def run(self, args: dict[str, Any]) -> str:
        query = args.get("query", "")
        results = self.retriever.retrieve(query, k=self.k)
        if not results:
            return "No relevant passages found."
        return "\n".join(f"[{r.chunk.chunk_id}] {r.chunk.text}" for r in results)


_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


class CalculatorTool(Tool):
    name = "calculate"
    description = "Evaluate a basic arithmetic expression safely. args: {expression: str}"

    def run(self, args: dict[str, Any]) -> str:
        expression = args.get("expression", "")
        try:
            tree = ast.parse(expression, mode="eval")
            result = _safe_eval(tree.body)
            return str(result)
        except Exception as exc:  # noqa: BLE001 - surfaced back to the agent as an observation
            return f"calculation_error: {exc}"


class WebLookupTool(Tool):
    """Stub for an external web-search tool.

    In an offline/sandboxed reference environment there is no live web
    call; this returns a clearly-labeled stub observation so the
    orchestrator's tool-selection logic and trace format are still fully
    exercised. Swapping in a real search API (e.g. Bing/Tavily/Serper) is
    a one-class change behind the same `Tool` interface.
    """

    name = "web_lookup"
    description = "Look up external information not present in the knowledge base. args: {query: str}"

    def run(self, args: dict[str, Any]) -> str:
        query = args.get("query", "")
        logger.info("tools.web_lookup_stub_called", extra={"query": query})
        return (
            f"[stub] Live web lookup is disabled in this offline reference environment. "
            f"Would have searched for: '{query}'."
        )
