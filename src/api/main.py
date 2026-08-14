"""FastAPI application entrypoint.

Persona: GenAI Developer.
JD requirement covered: "Develop, implement, and maintain APIs to
integrate GenAI models into applications and workflows."

Run locally with:
    uvicorn src.api.main:app --reload
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.middleware.monitoring import request_logging_middleware
from src.api.routes import agent, health, ingest, metrics, query
from src.api.state import build_app_state
from src.monitoring.logger import configure_logging

configure_logging()
logger = logging.getLogger("documind.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.documind = build_app_state()
    logger.info(
        "api.startup",
        extra={
            "embedding_provider": app.state.documind.embedding_provider.name,
            "vector_store": app.state.documind.vector_store.name,
            "llm_provider": app.state.documind.llm.name,
        },
    )
    yield


app = FastAPI(
    title="DocuMind AI",
    description="Enterprise Hybrid-RAG & Agentic Knowledge Platform — reference GenAI Developer project.",
    version="1.0.0",
    lifespan=lifespan,
)

app.middleware("http")(request_logging_middleware)

app.include_router(health.router, tags=["health"])
app.include_router(ingest.router, tags=["ingest"])
app.include_router(query.router, tags=["query"])
app.include_router(agent.router, tags=["agent"])
app.include_router(metrics.router, tags=["metrics"])


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc: StarletteHTTPException):
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail:
        return JSONResponse(status_code=exc.status_code, content=detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": "HTTP_ERROR", "message": str(detail)}},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": {"code": "VALIDATION_ERROR", "message": str(exc.errors())}},
    )
