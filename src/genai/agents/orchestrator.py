"""Agent orchestrator: plan -> act -> observe loop.

Persona: GenAI Developer.
JD requirement covered: "Conduct research and experiments to explore
innovative generative AI techniques such as AI agents ... and workflow
orchestration."

The planner is pluggable via the same `LLMProvider` interface used by the
RAG chain. With `LLM_PROVIDER=stub` (the offline default) planning uses a
transparent, inspectable heuristic: it splits the goal into sub-questions
on connective keywords ("and", "compare", "then", "vs"), issues a
`retrieve` step per sub-question, and finishes once it has gathered
enough context. This mirrors exactly what a cloud LLM planner (GPT-4 /
Claude / Gemini) would be asked to do via a structured "next action"
prompt — this module only changes the *decision process*, never the loop
mechanics or trace format, when a real planning LLM is swapped in.

Cost/safety control: `max_steps` hard-caps the loop (NFR-7, risk R-4).
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from src.genai.agents.tools import Tool
from src.genai.llm import LLMProvider

logger = logging.getLogger("documind.genai.agents.orchestrator")

_SPLIT_RE = re.compile(r"\b(?:and|compare|versus|vs\.?|then|summarize)\b", re.IGNORECASE)


@dataclass
class AgentStep:
    step: int
    action: str
    args: dict[str, Any]
    observation: str | None


@dataclass
class AgentResult:
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    steps_used: int = 0
    max_steps: int = 0
    latency_ms: float = 0.0


class AgentOrchestrator:
    def __init__(
        self,
        tools: list[Tool],
        llm: LLMProvider,
        max_steps: int = 6,
    ):
        self.tools = {t.name: t for t in tools}
        self.llm = llm
        self.max_steps = max_steps

    def _plan_subqueries(self, goal: str) -> list[str]:
        parts = [p.strip() for p in _SPLIT_RE.split(goal) if p.strip()]
        # Guard against a goal with no connective keywords -> single-step plan.
        return parts if len(parts) > 1 else [goal]

    def run(self, goal: str, max_steps: int | None = None) -> AgentResult:
        start = time.time()
        cap = max_steps or self.max_steps
        subqueries = self._plan_subqueries(goal)[: max(cap - 1, 1)]

        steps: list[AgentStep] = []
        observations: list[str] = []
        step_num = 0

        for sq in subqueries:
            step_num += 1
            if step_num > cap:
                break
            retrieve_tool = self.tools.get("retrieve")
            observation = retrieve_tool.run({"query": sq}) if retrieve_tool else "retrieve tool unavailable"
            steps.append(AgentStep(step=step_num, action="retrieve", args={"query": sq}, observation=observation))
            observations.append(observation)
            logger.info("agent.step", extra={"step": step_num, "action": "retrieve", "query": sq})

        # Optional calculation step if the goal looks numeric.
        if step_num < cap and re.search(r"\d+\s*[\+\-\*/]\s*\d+", goal):
            step_num += 1
            expr_match = re.search(r"[\d\.\s\+\-\*/\(\)]{3,}", goal)
            calc_tool = self.tools.get("calculate")
            if expr_match and calc_tool:
                observation = calc_tool.run({"expression": expr_match.group(0)})
                steps.append(
                    AgentStep(step=step_num, action="calculate", args={"expression": expr_match.group(0)}, observation=observation)
                )
                observations.append(f"calculation result: {observation}")

        # Finish: synthesize final answer from gathered observations using the
        # same grounded-generation approach as the RAG chain.
        step_num += 1
        context = "\n\n".join(observations) if observations else "No observations were gathered."
        synthesis_prompt = (
            f"CONTEXT:\n{context}\nQUESTION:\n{goal}\nANSWER:\n"
        )
        final_answer = self.llm.generate(synthesis_prompt)
        steps.append(AgentStep(step=step_num, action="finish", args={}, observation=None))

        result = AgentResult(
            answer=final_answer,
            steps=steps,
            steps_used=step_num,
            max_steps=cap,
            latency_ms=round((time.time() - start) * 1000, 1),
        )
        logger.info(
            "agent.run_complete",
            extra={"goal": goal, "steps_used": result.steps_used, "latency_ms": result.latency_ms},
        )
        return result
