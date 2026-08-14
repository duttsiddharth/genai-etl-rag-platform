from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.middleware.auth import require_api_key
from src.api.schemas import IngestRequest, IngestResponse
from src.etl.extract import SUPPORTED_EXTENSIONS

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
async def ingest(payload: IngestRequest, request: Request) -> IngestResponse:
    state = request.app.state.documind

    if payload.source_paths:
        paths = payload.source_paths
    else:
        source_dir = Path(state.default_source_dir)
        if not source_dir.exists():
            raise HTTPException(
                status_code=400,
                detail={"error": {"code": "SOURCE_DIR_MISSING", "message": f"{source_dir} does not exist"}},
            )
        paths = [str(p) for p in source_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not paths:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "NO_SOURCES", "message": "No source files to ingest."}},
        )

    summary = state.etl_pipeline.run(paths, force=payload.force)
    return IngestResponse(
        documents_processed=summary.documents_processed,
        chunks_indexed=summary.chunks_indexed,
        skipped_unchanged=summary.skipped_unchanged,
        manifest=summary.manifest,
        duration_seconds=summary.duration_seconds,
    )
