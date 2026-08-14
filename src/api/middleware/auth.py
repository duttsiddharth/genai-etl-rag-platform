"""Minimal API-key auth dependency.

Persona: GenAI Developer / Solution Architect.
Deliberately a stub (as documented in `docs/03_architecture_design.md`
§8, "Security Architecture") — swap for OAuth2/OIDC or cloud IAM auth in
a real production deployment. Kept as a real, enforced check (not a
no-op) so the integration tests can exercise the 401 path (IT-05).
"""
from __future__ import annotations

from fastapi import Header, HTTPException, Request


async def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    expected = request.app.state.documind.api_key
    if x_api_key != expected:
        raise HTTPException(
            status_code=401,
            detail={"error": {"code": "UNAUTHORIZED", "message": "Missing or invalid X-API-Key header."}},
        )
