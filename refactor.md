# modelO_kit Refactoring Guide

**Date:** Senior Code Review  
**Reviewer:** Senior Engineer  
**Status:** Ready for Implementation

---

## Executive Summary

This document identifies areas for improvement in the codebase following a comprehensive review. The codebase has solid foundations with good use of design patterns (Factory, Strategy, Facade), but there are opportunities to improve **encapsulation**, **reduce duplication**, **simplify complexity**, and **better leverage existing protocols**.

### Priority Matrix

| Priority | Area | Impact | Effort |
|----------|------|--------|--------|
| 🔴 High | Encapsulation violations in handlers | High | Low |
| 🔴 High | Dead code / unused abstractions | Medium | Low |
| 🟡 Medium | Duplicate formatting patterns | Medium | Medium |
| 🟡 Medium | OutputWriter mixed concerns | Medium | Medium |
| 🟢 Low | Singleton pattern smell in PromptService | Low | Low |
| 🟢 Low | Missing abstractions for web/content | Low | Medium |

---

## 1. Encapsulation Violations 🔴

### Problem

In `cli/handlers.py`, the `PaperHandler._summarize()` method directly accesses protected members:

```python
# handlers.py:84-85
provider, model = self._llm_service._provider, self._llm_service._model
llm = get_llm_service(provider, model)
```

This breaks encapsulation - handlers should not know about the internal structure of `LLMService`.

### Root Cause

`LLMService` already has public properties (`provider`, `model`), but the handler bypasses them. More importantly, handlers shouldn't need to create their own LLM instances - that's the service's job.

### Solution

**Option A (Quick Fix):** Use public properties:
```python
# Use existing public properties
provider, model = self._llm_service.provider, self._llm_service.model
llm = get_llm_service(provider, model)
```

**Option B (Recommended):** Add a method to `LLMService` to get the underlying LLM:
```python
# In llm_service.py
def get_llm(self) -> BaseLLMService:
    """Get the configured LLM instance."""
    if not self.is_configured:
        raise RuntimeError("LLM not configured")
    return self._llm

# In handlers.py
llm = self._llm_service.get_llm()
```

**Option C (Best - Dependency Injection):** Inject `PaperSummarizer` or a factory:
```python
# handlers.py
def _summarize(self, paper: Paper) -> PaperSummary:
    summarizer = self._llm_service.create_summarizer()  # New method
    return summarizer.summarize(paper)
```

### Files to Modify
- `src/modelo_kit/services/llm_service.py` - Add `get_llm()` or `create_summarizer()` method
- `src/modelo_kit/cli/handlers.py` - Use new methods instead of protected access

---

## 2. Dead Code and Unused Abstractions 🔴

### Problem

Several abstractions were created but are never used:

#### 2.1 `ContentMode` Enum (core/enums.py)

```python
class ContentMode(Enum):
    """Content source modes."""
    PAPER = "paper"
    OBSIDIAN = "obsidian"
    WEB = "web"
```

This enum is **never imported or used** anywhere in the codebase. The CLI menu uses string keys instead.

#### 2.2 `NoteRepository` Protocol (core/interfaces.py)

```python
@runtime_checkable
class NoteRepository(Protocol):
    def get_note(self, name: str) -> ObsidianNote | None: ...
    def notes(self) -> Iterator[ObsidianNote]: ...
    # etc.
```

`ObsidianVault` implements this interface but **nothing ever depends on the protocol**. The concrete class is always used directly.

#### 2.3 `SummaryFormatter` Protocol (core/interfaces.py)

```python
@runtime_checkable
class SummaryFormatter(Protocol):
    def format(self, title: str, content: str, metadata: dict[str, Any]) -> str: ...

class MarkdownFormatter:
    """Format summaries as Markdown."""
    def format(...) -> str: ...
```

`MarkdownFormatter` is defined but **never instantiated or used**.

### Solution

**Option A (Clean up):** Remove unused code:
- Delete `ContentMode` from `enums.py`
- Delete `NoteRepository`, `SummaryFormatter`, and `MarkdownFormatter` from `interfaces.py`

**Option B (Utilize):** Actually use these abstractions:
- Use `ContentMode` in menu selection and handler routing
- Type-hint functions that accept vaults as `NoteRepository`
- Use `SummaryFormatter` in `OutputWriter`

### Recommended Approach

For now, **delete the unused code**. If future requirements need these abstractions, they can be re-added with proper usage. Dead code increases maintenance burden.

### Files to Modify
- `src/modelo_kit/core/enums.py` - Remove `ContentMode` or integrate it
- `src/modelo_kit/core/interfaces.py` - Remove unused protocols or use them
- `src/modelo_kit/models/__init__.py` - Update exports if needed

---

## 3. Duplicate Formatting Patterns 🟡

### Problem

Multiple handlers have identical patterns for printing separators and headers:

```python
# handlers.py:118-119
print("\n" + "=" * 60)
print("EXECUTIVE SUMMARY")
print("=" * 60)

# handlers.py:227-229
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

# handlers.py:299-301
print("=" * 60)
print(f"FOLDER SUMMARY: {folder_path}")
print("=" * 60)

# handlers.py:391-393
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
```

### Solution

Add a utility function to `cli/prompts.py`:

```python
# cli/prompts.py
def print_section(title: str, char: str = "=", width: int = 60) -> None:
    """Print a section header."""
    print("\n" + char * width)
    print(title.upper())
    print(char * width)

# Usage in handlers
print_section("Executive Summary")
print_section(f"Folder Summary: {folder_path}")
```

### Files to Modify
- `src/modelo_kit/cli/prompts.py` - Add `print_section()` function
- `src/modelo_kit/cli/handlers.py` - Replace duplicate print statements

---

## 4. OutputWriter Mixed Concerns 🟡

### Problem

`OutputWriter` mixes static utility methods with instance methods:

```python
class OutputWriter:
    # Static utility methods
    @staticmethod
    def _get_model_suffix(...) -> str: ...
    
    @staticmethod
    def _sanitize_filename(...) -> str: ...
    
    @staticmethod
    def get_name_from_source(...) -> str: ...
    
    # Instance methods
    def save_summary(...): ...
    def save_content(...): ...
    def save_parsed_paper(...): ...  # Has 6 parameters (noqa)
```

The static methods are called both internally and externally (e.g., from `PaperHandler`), which suggests they should be public utilities.

### Solution

**Option A (Separate utilities):**
```python
# core/filename_utils.py
def get_model_suffix(provider: str, model: str) -> str: ...
def sanitize_filename(name: str) -> str: ...
def get_name_from_source(source: str) -> str: ...

# core/output.py
from modelo_kit.core.filename_utils import sanitize_filename, get_model_suffix

class OutputWriter:
    def save_summary(self, name: str, title: str, content: str, provider: str, model: str) -> None:
        suffix = get_model_suffix(provider, model)
        filename = sanitize_filename(f"{name}{suffix}_summary")
        # ...
```

**Option B (Make public methods):** Remove underscore prefix from methods intended for external use:
```python
class OutputWriter:
    @staticmethod
    def get_model_suffix(...) -> str: ...  # Now public
    
    @staticmethod
    def sanitize_filename(...) -> str: ...  # Now public
```

### Recommended

**Option A** is cleaner - separate pure utility functions from the writer class.

### Files to Modify
- Create `src/modelo_kit/core/filename_utils.py`
- Update `src/modelo_kit/core/output.py` to import utilities
- Update `src/modelo_kit/cli/handlers.py` to import from new location

---

## 5. save_parsed_paper Signature 🟡

### Problem

```python
def save_parsed_paper(
    self,
    name: str,
    title: str,
    source: str,
    pages: int,
    abstract: str,
    sections_content: str,
) -> None:  # noqa: PLR0913 (too many args)
```

This method takes 6 parameters, which is a code smell. The `noqa` comment suppresses the linting warning but doesn't fix the issue.

### Solution

Create a data class for paper metadata:

```python
# core/output.py or models/content.py
@dataclass
class ParsedPaperOutput:
    """Data for saving parsed paper output."""
    name: str
    title: str
    source: str
    pages: int
    abstract: str
    sections_content: str

# Updated method
def save_parsed_paper(self, paper_output: ParsedPaperOutput) -> None:
    ...
```

Or, since `Paper` already exists, just pass the `Paper` model directly:

```python
def save_parsed_paper(self, paper: Paper, name: str) -> None:
    """Save parsed paper to file."""
    sections_content = self._format_sections(paper.sections)
    path = self.output_dir / f"{self._sanitize_filename(name)}_parsed.txt"
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Title: {paper.title}\n")
        f.write(f"Source: {paper.source_path}\n")
        # ...
```

### Files to Modify
- `src/modelo_kit/core/output.py` - Refactor method signature
- `src/modelo_kit/cli/handlers.py` - Update call sites

---

## 6. PromptService Singleton Pattern 🟢

### Problem

```python
class PromptService:
    _instance: "PromptService | None" = None
    _prompts: dict[str, Any] = {}
    _loaded: bool = False

    def __new__(cls) -> "PromptService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

This singleton pattern has issues:
1. Class-level mutable state (`_prompts`, `_loaded`)
2. Makes testing harder (can't easily reset state)
3. Hidden dependency (services just call `PromptService()`)

### Solution

**Option A (Module-level caching):**
```python
# prompt_service.py
from functools import lru_cache

@lru_cache(maxsize=1)
def get_prompt_service() -> PromptService:
    return PromptService()

class PromptService:
    def __init__(self) -> None:
        self._prompts = self._load_prompts()
    
    def _load_prompts(self) -> dict[str, Any]:
        # Load once in constructor
        ...
```

**Option B (Dependency injection):**
```python
# Pass PromptService to classes that need it
class PaperSummarizer:
    def __init__(self, llm: BaseLLMService, prompts: PromptService, config: SummarizerConfig):
        self.llm = llm
        self.prompts = prompts
        self.config = config
```

### Recommended

For this codebase size, **Option A** (module-level caching with `@lru_cache`) is simpler and achieves the same result.

### Files to Modify
- `src/modelo_kit/services/prompt_service.py` - Refactor singleton pattern

---

## 7. Handler Factory Pattern 🟢

### Problem

The main menu uses a dictionary dispatch:

```python
# app.py (implied from context)
handlers = {
    "1": PaperHandler,
    "2": ObsidianHandler,
    "3": WebHandler,
}
```

And `ObsidianHandler.run()` also uses internal dispatch:

```python
actions = {
    "1": lambda: self._summarize_note(vault),
    "2": lambda: self._summarize_folder(vault),
    "3": lambda: self._list_notes(vault),
}
```

While functional, this could be more type-safe and extensible.

### Solution

Use the `ContentMode` enum (if kept) with a registry:

```python
# cli/handler_registry.py
from modelo_kit.core.enums import ContentMode
from modelo_kit.cli.handlers import BaseHandler, PaperHandler, ObsidianHandler, WebHandler

_HANDLERS: dict[ContentMode, type[BaseHandler]] = {
    ContentMode.PAPER: PaperHandler,
    ContentMode.OBSIDIAN: ObsidianHandler,
    ContentMode.WEB: WebHandler,
}

def get_handler(mode: ContentMode) -> BaseHandler:
    handler_class = _HANDLERS.get(mode)
    if handler_class is None:
        raise ValueError(f"No handler for mode: {mode}")
    return handler_class()
```

### Files to Create/Modify
- Create `src/modelo_kit/cli/handler_registry.py` (optional)
- Update `src/modelo_kit/cli/app.py` to use registry

---

## 8. WebService Inconsistency 🟢

### Problem

```python
class WebService:
    @staticmethod
    def get_web_content(url: str) -> str:
        loader = WebBaseLoader(url)
        docs = loader.load()
        if not docs:
            return ""
        return "\n\n".join(doc.page_content for doc in docs)
```

This is the only service that's entirely static. It doesn't match the pattern of other services (`ObsidianVault`, `LLMService`, `PromptService`).

### Solution

Either:
1. Keep it static (it's simple and stateless - this is fine)
2. Make it a proper class for consistency:

```python
class WebService:
    def __init__(self, timeout: int = 30) -> None:
        self._timeout = timeout
    
    def get_content(self, url: str) -> str:
        loader = WebBaseLoader(url)
        # ... with timeout support
```

### Recommendation

Keep it static for now - it's simple and doesn't benefit from state.

---

## 9. Improved Type Safety 🟢

### Problem

Several places use `str` where more specific types would help:

```python
# Various places
provider: str  # Could be LLMProvider enum
model: str     # Could be a validated type
path: str      # Could be Path
```

### Solution

Use the existing `LLMProvider` enum more consistently:

```python
# llm_service.py
from modelo_kit.core.enums import LLMProvider

class LLMService:
    def configure(self, provider: LLMProvider | str, model: str) -> None:
        if isinstance(provider, str):
            # Convert string to enum for validation
            provider = LLMProvider(provider)
        self._provider = provider
        # ...
```

---

## Implementation Checklist

### Phase 1: Quick Wins (1-2 hours)
- [ ] Fix protected member access in `PaperHandler._summarize()`
- [ ] Add `print_section()` utility to `cli/prompts.py`
- [ ] Replace duplicate print patterns in handlers
- [ ] Decide: Delete or use `ContentMode` enum

### Phase 2: Cleanup (2-4 hours)
- [ ] Remove or utilize `NoteRepository` and `SummaryFormatter`
- [ ] Extract filename utilities to separate module
- [ ] Make `OutputWriter._get_model_suffix` and `_sanitize_filename` public

### Phase 3: Architecture (4+ hours)
- [ ] Refactor `save_parsed_paper` signature
- [ ] Add `get_llm()` method to `LLMService`
- [ ] Consider refactoring `PromptService` singleton

---

## Testing Notes

After implementing changes:

```bash
# Run type checking
mypy src/

# Run linting
ruff check src/

# Run any existing tests
pytest tests/  # if exists
```

---

## Appendix: Files Changed in This Review

| File | Lines | Issues Found |
|------|-------|--------------|
| `cli/handlers.py` | 415 | Encapsulation, duplication |
| `core/enums.py` | 50 | Unused ContentMode |
| `core/interfaces.py` | 65 | Unused protocols |
| `core/output.py` | 186 | Mixed concerns, long signature |
| `services/llm_service.py` | 130 | Could expose get_llm() |
| `services/prompt_service.py` | 78 | Singleton pattern |

---

*Document generated by Senior Engineer Code Review*
