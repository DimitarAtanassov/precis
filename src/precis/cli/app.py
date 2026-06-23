"""Précis command-line interface (Typer).

A thin client over the same application services and composition root the API
uses — one source of truth, two surfaces. Async services are bridged to the
synchronous CLI boundary with ``asyncio.run``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from precis.config.settings import DEFAULT_MODEL, DEFAULT_PROVIDER
from precis.container import Container, build_container
from precis.domain.errors import PrecisError
from precis.llm_factory import get_llm_service
from precis.parsers import PaperParser
from precis.services.summarizer_service import PaperSummarizer, SummarizerConfig
from precis.services.web_service import WebService

app = typer.Typer(
    help="Précis — parse and summarize content with LLMs.",
    no_args_is_help=True,
    add_completion=False,
)
summarize_app = typer.Typer(help="Summarize content from various sources.")
app.add_typer(summarize_app, name="summarize")

console = Console()
err_console = Console(stderr=True)

ProviderOpt = Annotated[str, typer.Option(help="LLM provider id.")]
ModelOpt = Annotated[str, typer.Option(help="Provider model name.")]


async def _summarize_text(
    container: Container, content: str, provider: str, model: str, prompt_key: str
) -> str:
    """Resolve a provider and return an LLM summary of ``content``."""
    llm = get_llm_service(provider, model, container.settings)
    prompt = container.prompt_service().get(prompt_key, content=content)
    if prompt.system_prompt:
        llm.set_system_prompt(prompt.system_prompt)
    return await llm.ask(prompt.user_prompt)


def _fail(message: str) -> None:
    err_console.print(f"[bold red]Error:[/] {message}")
    raise typer.Exit(code=1)


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind host.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Bind port.")] = 8000,
    reload: Annotated[bool, typer.Option(help="Auto-reload on changes.")] = False,
) -> None:
    """Run the HTTP API with uvicorn."""
    import uvicorn  # noqa: PLC0415 - optional/heavy import, only needed to serve

    uvicorn.run("precis.api.main:app", host=host, port=port, reload=reload)


@app.command()
def prompts() -> None:
    """List available prompt keys."""
    container = build_container()
    for key in container.prompt_service().list_keys():
        console.print(f"• {key}")


@summarize_app.command("text")
def summarize_text_cmd(
    text: Annotated[str | None, typer.Argument(help="Text to summarize.")] = None,
    file: Annotated[
        Path | None, typer.Option(help="Read text from a file instead.")
    ] = None,
    provider: ProviderOpt = DEFAULT_PROVIDER,
    model: ModelOpt = DEFAULT_MODEL,
) -> None:
    """Summarize a block of text (positional) or a text file (--file)."""
    if file is not None:
        content = file.read_text(encoding="utf-8")
    elif text is not None:
        content = text
    else:
        _fail("Provide TEXT or --file.")
        return

    container = build_container()
    try:
        summary = asyncio.run(
            _summarize_text(container, content, provider, model, "webpage_summary")
        )
    except PrecisError as exc:
        _fail(str(exc))
        return
    console.print(summary)


@summarize_app.command("web")
def summarize_web_cmd(
    url: Annotated[str, typer.Argument(help="Web page URL.")],
    provider: ProviderOpt = DEFAULT_PROVIDER,
    model: ModelOpt = DEFAULT_MODEL,
) -> None:
    """Fetch a web page and summarize it."""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    content = WebService.get_web_content(url)
    if not content:
        _fail(f"Could not fetch content from {url}.")
        return

    container = build_container()
    try:
        summary = asyncio.run(
            _summarize_text(container, content, provider, model, "webpage_summary")
        )
    except PrecisError as exc:
        _fail(str(exc))
        return
    console.print(summary)


@summarize_app.command("paper")
def summarize_paper_cmd(
    source: Annotated[str, typer.Argument(help="PDF path or URL.")],
    provider: ProviderOpt = DEFAULT_PROVIDER,
    model: ModelOpt = DEFAULT_MODEL,
) -> None:
    """Parse a research paper (PDF path or URL) and summarize it."""
    container = build_container()
    paper = PaperParser().parse(source, load_content=True)

    llm = get_llm_service(provider, model, container.settings)
    config = SummarizerConfig.for_model(model, verbose=True)
    try:
        summary = asyncio.run(
            PaperSummarizer(llm, config, container.prompt_service()).summarize(paper)
        )
    except PrecisError as exc:
        _fail(str(exc))
        return

    path = container.output_writer().save_summary(
        name=paper.title,
        title=summary.title,
        content=summary.to_markdown(),
        provider=provider,
        model=model,
    )
    console.print(f"[bold]Executive Summary[/]\n{summary.executive_summary}")
    console.print(f"\n[green]Saved to[/] {path}")


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
