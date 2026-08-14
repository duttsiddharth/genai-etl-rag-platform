from __future__ import annotations

from fastapi import APIRouter, Request

from src.api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    state = request.app.state.documind
    return HealthResponse(
        status="ok",
        vector_store=state.vector_store.name,
        documents_indexed=state.vector_store.count(),
    )
