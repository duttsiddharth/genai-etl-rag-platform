from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from src.api.middleware.auth import require_api_key
from src.api.schemas import CitationOut, QueryRequest, QueryResponse
from src.monitoring.metrics import LOW_CONFIDENCE_ANSWERS_TOTAL, TOKENS_USED_TOTAL

router = APIRouter()

_LOW_CONFIDENCE_THRESHOLD = 0.05


@router.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
async def query(payload: QueryRequest, request: Request) -> QueryResponse:
    state = request.app.state.documind

    result = state.rag_chain.answer(payload.question, k=payload.k, alpha=payload.alpha)

    if not result.citations or all(c["score"] < _LOW_CONFIDENCE_THRESHOLD for c in result.citations):
        LOW_CONFIDENCE_ANSWERS_TOTAL.inc()

    TOKENS_USED_TOTAL.inc(len(result.answer.split()))  # word-count proxy in the offline stub

    return QueryResponse(
        answer=result.answer,
        citations=[CitationOut(**c) for c in result.citations],
        retrieval_strategy=result.retrieval_strategy,
        latency_ms=result.latency_ms,
    )
