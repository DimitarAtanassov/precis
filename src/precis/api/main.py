"""FastAPI application factory.

`create_app` builds the composition root once and exposes it on ``app.state`` so
routes resolve fully-wired services via dependencies. A module-level ``app`` is
provided for ``uvicorn precis.api.main:app``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from precis.api.errors import install_exception_handlers
from precis.api.middleware import RequestIDMiddleware
from precis.api.routes import health, prompts, summarize
from precis.container import Container, build_container
from precis.observability import configure_tracing, get_logger, instrument_fastapi

_logger = get_logger(__name__)


def create_app(container: Container | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    container = container or build_container()
    configure_tracing(container.settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        _logger.info("api_startup", prompt_source=container.settings.prompt_source)
        yield
        _logger.info("api_shutdown")

    app = FastAPI(
        title="Précis API",
        version="0.1.0",
        summary="Parse and summarize content with LLMs.",
        lifespan=lifespan,
    )
    app.state.container = container

    app.add_middleware(RequestIDMiddleware)
    install_exception_handlers(app)
    instrument_fastapi(app, container.settings)

    app.include_router(health.router)
    app.include_router(prompts.router, prefix="/v1")
    app.include_router(summarize.router, prefix="/v1")
    return app


app = create_app()
