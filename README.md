# 🧠 Précis

A Python toolkit for parsing and summarizing content using LLMs. Works with research papers (PDFs), Obsidian vaults, and web pages.

Built with clean architecture principles—easy to understand, extend, and adapt to your needs.

## What It Does

| Mode | Input | Output |
|------|-------|--------|
| 📄 **Paper** | PDF file or URL | Structured sections + LLM summary |
| 📓 **Obsidian** | Vault path | Note summaries, folder overviews |
| 🌐 **Web** | Any URL | Page content + LLM summary |

**Supported LLM Providers:** Claude, OpenAI, Gemini, DeepSeek

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/DimitarAtanassov/precis.git
cd precis
uv sync  # or: pip install -e .

# Set up API keys
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
# Add others as needed: OPENAI_API_KEY, GOOGLE_API_KEY, DEEPSEEK_API_KEY

# Run
uv run precis
```

---

## Architecture Overview

```
src/precis/
├── cli/                    # 🎯 Presentation Layer
│   ├── app.py              #    Entry point & main menu
│   ├── handlers.py         #    Mode handlers (Paper, Obsidian, Web)
│   ├── menu.py             #    Menu display utilities
│   └── prompts.py          #    User input helpers
│
├── core/                   # 🔧 Foundation Layer
│   ├── enums.py            #    LLMProvider enum
│   ├── filename_utils.py   #    Filename sanitization
│   ├── interfaces.py       #    Protocols (contracts)
│   └── output.py           #    File writing
│
├── llms/                   # 🤖 LLM Abstraction Layer
│   ├── llm_base.py         #    Base class (all shared logic)
│   ├── llm_claude.py       #    Claude provider (~35 lines)
│   ├── llm_openai.py       #    OpenAI provider (~35 lines)
│   ├── llm_gemini.py       #    Gemini provider (~35 lines)
│   └── llm_deepseek.py     #    DeepSeek provider (~40 lines)
│
├── models/                 # 📦 Data Models
│   ├── content.py          #    Paper, Section, ObsidianNote
│   └── llm.py              #    Prompt, LLMOutput, Summary
│
├── parsers/                # 📄 Content Extraction
│   ├── base.py             #    BaseParser ABC
│   ├── pdf.py              #    PDF parsing strategies
│   ├── markdown.py         #    Obsidian markdown parser
│   └── paper.py            #    PaperParser facade
│
├── services/               # ⚙️ Business Logic
│   ├── llm_service.py      #    High-level LLM orchestration
│   ├── obsidian_vault.py   #    Vault discovery & filtering
│   ├── prompt_service.py   #    YAML prompt loading
│   ├── summarizer_service.py  # Paper summarization pipeline
│   └── web_service.py      #    Web content fetching
│
├── llm_factory.py          # 🏭 Factory function
└── prompts.yaml            # 📝 All prompt templates
```

---

## Design Patterns & Why

### 1. Factory Pattern (`llm_factory.py`)

**Problem:** We need to create different LLM providers without the caller knowing which class to instantiate.

**Solution:** A simple factory function maps provider names to classes:

```python
from precis.llm_factory import get_llm_service

# Caller doesn't need to know about ClaudeLLMService, OpenAILLMService, etc.
llm = get_llm_service("claude", "claude-sonnet-4-5-20250929")
```

**Why this helps:** Adding a new provider = add one class + one line in the factory. No changes to calling code.

### 2. Template Method Pattern (`llms/llm_base.py`)

**Problem:** All 4 LLM providers share 90% identical code (retry logic, message building, fallback parsing).

**Solution:** `BaseLLMService` implements all shared logic. Subclasses only define initialization:

```python
class ClaudeLLMService(BaseLLMService):
    def _init_chat(self) -> BaseChatModel:
        return ChatAnthropic(api_key=..., model_name=self.model_name)
    
    @property
    def provider_name(self) -> str:
        return "Claude"
```

**Result:** Each provider file is ~35 lines instead of ~130 lines. DRY principle in action.

### 3. Strategy Pattern (`parsers/pdf.py`)

**Problem:** PDFs can be parsed different ways—by embedded TOC or by analyzing fonts. We need to pick the best one at runtime.

**Solution:** Two strategies implement `BaseParser`:
- `PDFTocParser` — Uses embedded outline (fast, accurate when available)
- `PDFFontParser` — Analyzes font sizes to detect headings (fallback)

```python
class PaperParser:  # Facade
    def __init__(self):
        self._parsers = [PDFTocParser(), PDFFontParser()]
    
    def parse(self, source):
        for parser in self._parsers:
            if parser.can_parse(source):
                return parser.parse(source)
```

### 4. Facade Pattern (`parsers/paper.py`)

**Problem:** Parsing papers involves multiple steps—downloading, selecting strategy, loading content. Callers shouldn't manage this.

**Solution:** `PaperParser` provides a simple interface:

```python
parser = PaperParser()
paper = parser.parse("https://arxiv.org/pdf/1234.5678.pdf")  # That's it
```

Behind the scenes: downloads PDF, tries TOC parser, falls back to font parser, handles URLs vs local files.

### 5. Command Pattern (`cli/handlers.py`)

**Problem:** Each CLI mode (Paper, Obsidian, Web) has different workflows but shares common setup (LLM configuration, output writing).

**Solution:** `BaseHandler` defines the structure; concrete handlers implement `run()`:

```python
class BaseHandler(ABC):
    def __init__(self):
        self._llm_service = LLMService()
        self._output = OutputWriter()
    
    @abstractmethod
    def run(self) -> None: ...

class PaperHandler(BaseHandler):
    def run(self):
        # Paper-specific workflow
```

---

## How to Extend

### Add a New LLM Provider

1. Create `llms/llm_newprovider.py`:

```python
from precis.llms.llm_base import BaseLLMService

class NewProviderLLMService(BaseLLMService):
    def __init__(self, model_name: str = "default-model") -> None:
        super().__init__(model_name)
        self.chat = self._init_chat()

    @property
    def provider_name(self) -> str:
        return "NewProvider"

    def _init_chat(self) -> BaseChatModel:
        api_key = os.getenv("NEWPROVIDER_API_KEY")
        return SomeLangChainChat(api_key=api_key, model=self.model_name)
```

2. Register in `llm_factory.py`:

```python
_PROVIDERS["newprovider"] = NewProviderLLMService
```

3. Add to menu in `core/enums.py` (optional).

### Add a New Content Mode

1. Create a handler in `cli/handlers.py`:

```python
class NewModeHandler(BaseHandler):
    def run(self) -> None:
        # Your workflow here
        pass
```

2. Add to the main menu in `cli/app.py`.

### Add a New Prompt Template

Edit `prompts.yaml`:

```yaml
my_new_prompt:
  system_prompt:
    prompt: "You are a helpful assistant..."
  user_prompt:
    prompt: "Please analyze: {content}"
```

Use it:

```python
from precis.services.prompt_service import PromptService

prompts = PromptService()
prompt = prompts.get("my_new_prompt", content="Hello world")
```

---

## Usage Examples

### CLI Mode

```bash
uv run precis
```

```
🧠 Précis - Content Parser & Summarizer
==================================================

Select content source:
  1. 📄 Research Paper (PDF file or URL)
  2. 📓 Obsidian Vault (markdown notes)
  3. 🌐 Web Page (URL)
  4. ❌ Exit
```

### Programmatic Usage

```python
# Parse a paper
from precis.parsers import PaperParser

parser = PaperParser()
paper = parser.parse("paper.pdf", load_content=True)
print(f"Title: {paper.title}")
print(f"Sections: {len(paper.sections)}")

# Browse Obsidian vault
from precis.services.obsidian_vault import ObsidianVault

vault = ObsidianVault("~/Documents/MyVault")
for note in vault.notes_with_tag("project"):
    print(f"- {note.title} ({note.word_count} words)")

# Use LLM directly
from precis.llm_factory import get_llm_service

llm = get_llm_service("claude", "claude-sonnet-4-5-20250929")
response = llm.ask("Summarize quantum computing in 3 sentences.")

# Structured output
from pydantic import BaseModel

class Analysis(BaseModel):
    summary: str
    key_points: list[str]

result = llm.ask_structured("Analyze this text...", Analysis)
print(result.key_points)
```

---

## Development

```bash
# Lint & format
uv run ruff check src/ --fix
uv run ruff format src/

# Type check
uv run mypy src/

# Run tests
uv run pytest
```

### Code Quality

- **Ruff** for linting (fast, replaces flake8/isort/pyupgrade)
- **MyPy strict** for type checking
- **Pydantic** for validated data models

---

## Key Files to Know

| When you want to... | Look at... |
|---------------------|------------|
| Add a new LLM provider | `llms/llm_base.py`, `llm_factory.py` |
| Change prompt templates | `prompts.yaml` |
| Modify CLI flow | `cli/handlers.py` |
| Understand PDF parsing | `parsers/paper.py`, `parsers/pdf.py` |
| See data structures | `models/content.py` |
| Change output format | `core/output.py` |

---

## License

MIT