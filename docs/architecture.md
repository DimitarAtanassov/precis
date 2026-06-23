# Architecture

This doc explains how the code is organized and why. The grammar is simple on
purpose.

## The big idea

Précis uses **hexagonal architecture** (also called "ports and adapters").

The idea is simple:

- The **core** (business logic) does not know about the outside world.
- The **outside world** (HTTP, the CLI, LLM SDKs, files) plugs into the core.
- They meet at **ports**. A port is just a Python `Protocol` (an interface).

This keeps the core easy to test and easy to change. You can swap an LLM
provider, a prompt source, or even the whole API without touching the core.

## The layers

```
inbound adapters         core                     outbound adapters
┌───────────────┐    ┌──────────────────┐    ┌───────────────────────┐
│ api/ (FastAPI)│──▶ │ services/         │──▶ │ llms/   (LLM SDKs)    │
│ cli/ (Typer)  │    │ orchestration/    │    │ parsers/, web, files  │
└───────────────┘    └──────────────────┘    └───────────────────────┘
                     depends only on ▼
              ports/ (Protocols)   domain/ (models + errors)
```

Read the layers from the inside out:

| Folder | What it is | Knows about |
|---|---|---|
| `domain/` | Plain data models + error types. No I/O. | Nothing |
| `ports/` | Interfaces (`LLMProvider`, `PromptRepository`, `TokenCounter`). | `domain/` |
| `config/` | `Settings` from env vars (one source of truth). | `domain/` |
| `orchestration/` | The agent layer: `Step`, `Pipeline`, `ToolRegistry`. | `domain/` |
| `services/` | The use-cases (e.g. summarize a paper). | ports, domain |
| `llms/` | LLM provider adapters (Claude, OpenAI, ...). | ports, SDKs |
| `parsers/` | Turn PDFs and markdown into models. | domain |
| `observability/` | Logging, metrics, tracing. | config |
| `api/` | The HTTP service (FastAPI). | services |
| `cli/` | The command-line tool (Typer). | services |
| `container.py` | Wires everything together. | all of the above |

**Rule of thumb:** arrows point inward. `api/` and `cli/` depend on `services/`.
`services/` depends on `ports/`. Nothing depends on `api/` or `cli/`.

## Key pieces

### Ports (the interfaces)

`ports/` holds Protocols. The most important is `LLMProvider`:

```python
class LLMProvider(Protocol):
    async def ask(self, prompt: str) -> str: ...
    async def ask_structured(self, prompt, output_schema, system_prompt=None): ...
```

Services depend on this Protocol, not on Claude or OpenAI directly. So you can
pass a real provider in production and a fake one in tests.

### The composition root (`container.py`)

This is the one place that builds real objects from settings:

```python
container = build_container()        # reads settings, sets up logging
prompts = container.prompt_service() # the prompt source
llm = container.llm_service()        # the LLM service
```

Both the API and the CLI build a container and ask it for services. They never
`new` up dependencies by hand. This makes wiring explicit and testable.

### The orchestration / agent layer

We built our own small agent layer. We did **not** use a framework like
LangGraph. It has three parts:

- **`Step`** — one unit of work. It takes a context and returns it.
- **`Pipeline`** — runs steps in order. It logs, times, and wraps errors.
- **`ToolRegistry`** — a place to register tools that agents can call later.

The paper summarizer is a `Pipeline` of steps:

```
SummarizeSectionsStep → ExecutiveSummaryStep → ContributionsStep
```

To add a new step, write a small class and add it to the pipeline. To add a
tool-using agent later, register tools and let a step pick which to call.

### Async core

All LLM calls are `async`. The API is async end-to-end. The CLI bridges to async
with `asyncio.run(...)` at the edge. Async lets us run many calls at once (for
example, summarizing sections in parallel — see `section_concurrency`).

## How a request flows (API)

`POST /v1/summarize/text`:

1. Middleware adds an `X-Request-ID` to every log line for that request.
2. The route asks the container for the prompt source and an LLM provider.
3. It renders the prompt and calls the provider (`await provider.ask(...)`).
4. If anything fails with a `PrecisError`, the API returns an RFC 7807
   `application/problem+json` body with the right status code.

## Errors

All our errors inherit from `PrecisError` (in `domain/errors.py`). This lets the
API map them to clean HTTP responses in one place:

| Error | HTTP status |
|---|---|
| `PromptNotFoundError` | 404 |
| `ParseError` | 422 |
| `LLMError` / `StructuredOutputError` | 502 |
| `ConfigError`, `PipelineError`, other | 500 |

## How to extend it

| You want to... | Do this |
|---|---|
| Add an LLM provider | Add an adapter in `llms/`, register it in `llm_factory`. |
| Add a prompt source | Implement `PromptRepository`, wire it in `container.py`. |
| Add a pipeline step | Implement `Step`, add it to a `Pipeline`. |
| Add an endpoint | Add a route in `api/routes/`, reuse the services. |
| Add a CLI command | Add a command in `cli/app.py`. |

## See also

- [technical-leadership.md](technical-leadership.md) — the decisions and why.
- [database.md](database.md) — the prompt store.
