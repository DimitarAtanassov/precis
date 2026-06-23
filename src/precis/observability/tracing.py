"""Optional OpenTelemetry tracing.

Tracing is an opt-in, enterprise-ready hook: enabled via settings and only
active when the OpenTelemetry packages are installed (``pip install
'precis[otel]'``). Everything here degrades to a safe no-op otherwise, so the
core never hard-depends on a tracing backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from precis.observability.logging import get_logger

if TYPE_CHECKING:
    from fastapi import FastAPI

    from precis.config.settings import Settings

_logger = get_logger(__name__)

try:  # pragma: no cover - exercised only when the extra is installed
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _OTEL_AVAILABLE = False


def configure_tracing(settings: Settings) -> None:
    """Configure a global tracer provider + OTLP exporter, if enabled."""
    if not settings.otel_enabled:
        return
    if not _OTEL_AVAILABLE:
        _logger.warning("otel_unavailable", hint="install precis[otel]")
        return

    resource = Resource.create({"service.name": settings.service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=settings.otel_endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _logger.info("otel_configured", endpoint=settings.otel_endpoint)


def instrument_fastapi(app: FastAPI, settings: Settings) -> None:
    """Instrument a FastAPI app for tracing, if enabled and available."""
    if not settings.otel_enabled or not _OTEL_AVAILABLE:
        return
    try:  # pragma: no cover - requires the extra
        from opentelemetry.instrumentation.fastapi import (  # noqa: PLC0415
            FastAPIInstrumentor,
        )

        FastAPIInstrumentor.instrument_app(app)
        _logger.info("otel_fastapi_instrumented")
    except ImportError:  # pragma: no cover
        _logger.warning("otel_fastapi_unavailable")
