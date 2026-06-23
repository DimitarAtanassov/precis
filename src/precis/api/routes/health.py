"""Liveness and readiness probes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from precis.api.dependencies import PromptsDep
from precis.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    """Liveness: the process is up and serving."""
    return HealthResponse(status="ok")


@router.get("/readyz", response_model=HealthResponse)
async def readyz(prompts: PromptsDep) -> HealthResponse:
    """Readiness: dependencies (the prompt source) are reachable."""
    try:
        prompts.list_keys()
    except Exception as exc:
        # Any failure reaching the prompt source means we are not ready.
        raise HTTPException(
            status_code=503, detail="prompt source unavailable"
        ) from exc
    return HealthResponse(status="ready")
