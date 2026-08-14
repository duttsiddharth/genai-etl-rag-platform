from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from src.api.middleware.auth import require_api_key
from src.api.schemas import AgentRunRequest, AgentRunResponse, AgentStepOut
from src.monitoring.metrics import AGENT_STEPS_USED

router = APIRouter()


@router.post("/agent/run", response_model=AgentRunResponse, dependencies=[Depends(require_api_key)])
async def agent_run(payload: AgentRunRequest, request: Request) -> AgentRunResponse:
    state = request.app.state.documind

    result = state.agent.run(payload.goal, max_steps=payload.max_steps)
    AGENT_STEPS_USED.observe(result.steps_used)

    return AgentRunResponse(
        answer=result.answer,
        steps=[AgentStepOut(**s.__dict__) for s in result.steps],
        steps_used=result.steps_used,
        max_steps=result.max_steps,
        latency_ms=result.latency_ms,
    )
