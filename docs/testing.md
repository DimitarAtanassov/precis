# Testing and the Quality Gate

This doc shows how to check your work. The grammar is simple on purpose.

## The quality gate

Four checks must pass. CI runs all of them. So should you, before you push.

```bash
uv run ruff check src tests       # 1. lint (style + common bugs)
uv run ruff format --check src tests  # 2. formatting
uv run mypy src                   # 3. types (strict mode)
uv run pytest --cov=precis        # 4. tests + coverage
```

To fix formatting and easy lint issues automatically:

```bash
uv run ruff format src tests
uv run ruff check --fix src tests
```

## Tests

Tests live in `tests/`. They are fast. They do **not** call real LLMs or the
network.

### The fake LLM

We never call a real model in tests. We use `FakeLLMService` (in
`tests/conftest.py`). It returns canned answers. So tests are fast and free.

You get it as a fixture:

```python
async def test_something(fake_llm):
    result = await fake_llm.ask("hi")
    assert result.startswith("FAKE_ANSWER::")
```

### Testing the API

We use FastAPI's `TestClient`. We swap the real LLM for the fake one with
`dependency_overrides`. No network is touched.

```python
app.dependency_overrides[get_provider_factory] = lambda: (lambda p, m: fake_llm)
client = TestClient(app)
assert client.get("/healthz").status_code == 200
```

### Testing the CLI

We use Typer's `CliRunner`. We patch the provider with the fake.

```python
runner.invoke(app, ["prompts"])
```

## Coverage gate (the "ratchet")

We measure how much code the tests cover. There is a minimum, set in
`pyproject.toml`:

```toml
fail_under = 58
```

The project started with **zero** tests. So we did not demand 100% on day one.
We set a real floor and raise it as we add tests. If you add code, add tests, and
nudge the floor up.

## Characterization tests

Some tests pin the **current** behavior of old code (for example, the paper
summarizer). They are a safety net. When we refactor, these must stay green. If
one breaks during a refactor, the behavior changed — check if that was on
purpose.

## Pre-commit hooks (optional but nice)

Run the gate automatically before each commit:

```bash
uv run pre-commit install
```

Now `ruff`, `mypy`, and `pytest` run when you commit.

## Continuous integration

`.github/workflows/ci.yml` runs the same gate on every push and pull request.
Keep it green.
