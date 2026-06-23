"""Prompt catalog endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from precis.api.dependencies import PromptsDep
from precis.api.schemas import PromptsResponse

router = APIRouter(tags=["prompts"])


@router.get("/prompts", response_model=PromptsResponse)
async def list_prompts(prompts: PromptsDep) -> PromptsResponse:
    """List the available prompt keys."""
    return PromptsResponse(keys=prompts.list_keys())
