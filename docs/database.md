# Database and the Prompt Store

This doc explains where prompts come from. The grammar is simple on purpose.

## Short version

- Précis needs prompts to talk to LLMs.
- By default, prompts come from a **YAML file**. No database is needed.
- You can switch to a **Postgres database** for prompts. This is optional.

So: **you do not need a database to run Précis.** Read on only if you want the
database, or you want to understand the design.

## Two prompt sources, one interface

Both sources hide behind one port: `PromptRepository` (in `ports/prompts.py`).

```python
class PromptRepository(Protocol):
    def get(self, key: str, **kwargs) -> Prompt: ...   # render a prompt
    def exists(self, key: str) -> bool: ...
    def list_keys(self) -> list[str]: ...
```

The rest of the code only knows this interface. It does not care if the prompt
came from a file or a database. You pick the source with one setting:

```bash
PRECIS_PROMPT_SOURCE=yaml       # default
PRECIS_PROMPT_SOURCE=database   # optional
```

## Source 1: YAML (default)

The prompts live in `src/precis/prompts.yaml`. Each prompt has a name and a
template. There are three shapes:

```yaml
# 1. Full: separate system and user prompts
paper_executive_summary:
  system_prompt:
    prompt: "You are an expert reviewer."
  user_prompt:
    prompt: "Summarize {paper_title}. Sections: {section_summaries}"

# 2. Simple: just a user prompt
webpage_summary:
  prompt: "Summarize this page: {content}"

# 3. Direct string
quick:
  "Summarize: {content}"
```

`{name}` placeholders are filled in at call time with Python's `str.format`.

**Why YAML:** It works offline. It is easy to edit. It is perfect for local dev
and tests.

## Source 2: Postgres database (optional)

For a large team, you may want prompts in a database. Then you can:

- Change prompts without a code deploy.
- Keep many **versions** of a prompt.
- Share prompts across services.

Précis reads database prompts through an external package called
**`floating_prompts`**. Précis does not own the database schema.
`floating_prompts` does. Précis only reads prompts by name (and optional
version), then fills in the variables.

### How the read works

```
precis  ──get("paper_summary", version=3)──▶  floating_prompts  ──SQL──▶  Postgres
        ◀──Prompt(system, user)──────────────                  ◀──row──
```

The table holds, in plain terms:

| Column (conceptual) | Meaning |
|---|---|
| `name` | The prompt key, e.g. `paper_summary`. |
| `version` | A number. Lets you keep history and roll back. |
| `system_prompt` | The system text (may be empty). |
| `user_prompt` | The user text with `{placeholders}`. |

Note: the exact schema lives in the `floating_prompts` project, not here.

### How to enable the database

1. Install `floating_prompts` (it is not a default dependency).
2. Set the prompt source:

   ```bash
   PRECIS_PROMPT_SOURCE=database
   PRECIS_PROMPT_VERSION=3        # optional; leave unset for the latest
   ```

3. Give `floating_prompts` its database connection (it reads its own env vars).
4. Start Précis as normal.

### Caching

For high traffic, wrap any source with `CachedPromptProvider`
(`services/prompt_provider.py`). It keeps prompts in memory for a short time
(TTL). This cuts database calls.

## Current status (important)

The `floating_prompts` package is being restructured right now. Its public API
changed. So the database provider needs a small repair before it works again.

**Until then, use the YAML source (the default).** It is fully working.

Précis is built so this does not block you:

- `floating_prompts` is **not** a required dependency.
- The database provider imports it **lazily**. It only loads when you choose the
  database source. If it is missing, you get a clear error — and only then.

## Postgres in Docker

`docker compose up` starts Postgres next to the API, ready for when the database
provider is enabled.

| Setting | Value |
|---|---|
| Image | `postgres:17-alpine` |
| User / Password / DB | `precis` / `precis` / `precis` |
| Port | `5432` |
| Data volume | `pgdata` (data survives restarts) |

The API still defaults to YAML, so it runs even if you do not use Postgres.

## See also

- [architecture.md](architecture.md) — how the ports fit together.
- `src/precis/services/prompt_provider.py` — the YAML, database, and cache code.
- `src/precis/services/prompt_service.py` — picks the source from settings.
