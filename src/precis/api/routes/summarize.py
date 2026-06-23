"""Summarization endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from precis.api.dependencies import PromptsDep, ProviderFactoryDep
from precis.api.schemas import SummarizeTextRequest, SummaryResponse

router = APIRouter(tags=["summarize"], prefix="/summarize")


@router.post("/text", response_model=SummaryResponse)
async def summarize_text(
    body: SummarizeTextRequest,
    prompts: PromptsDep,
    provider_factory: ProviderFactoryDep,
) -> SummaryResponse:
    """Summarize an arbitrary block of text with the chosen provider/model."""
    provider = provider_factory(body.provider, body.model)
    prompt = prompts.get(body.prompt_key, content=body.text)
    if prompt.system_prompt:
        provider.set_system_prompt(prompt.system_prompt)
    summary = await provider.ask(prompt.user_prompt)
    return SummaryResponse(summary=summary, provider=body.provider, model=body.model)
