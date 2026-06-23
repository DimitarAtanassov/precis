"""Domain error hierarchy.

A single rooted exception tree (`PrecisError`) lets every caller — CLI, API
exception handlers, tests — catch precis failures distinctly from arbitrary
third-party exceptions, and lets the API map them to stable HTTP problem
responses without leaking adapter internals.
"""

from __future__ import annotations


class PrecisError(Exception):
    """Base class for all precis errors."""


class ConfigError(PrecisError):
    """Raised when configuration is missing or invalid (e.g. an absent API key)."""


class LLMError(PrecisError):
    """Base class for failures originating from an LLM provider."""


class StructuredOutputError(LLMError):
    """Raised when an LLM fails to produce parseable structured output."""


class PromptNotFoundError(PrecisError, KeyError):
    """Raised when a prompt key cannot be resolved from a prompt source.

    Subclasses ``KeyError`` for backward compatibility with existing callers
    that catch ``KeyError`` from prompt providers.
    """


class ParseError(PrecisError):
    """Raised when source content (PDF, markdown, web) cannot be parsed."""


class PipelineError(PrecisError):
    """Raised when an orchestration step fails unexpectedly.

    Wraps the originating exception so the failing step is identifiable while
    preserving the hexagon's single error root.
    """
