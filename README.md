# 🧠 Modelo Kit

A Python toolkit for parsing and summarizing content from research papers, Obsidian vaults, and web pages using LLMs.

## Features

- **📄 Research Papers** - Parse PDFs (local or URL) into structured sections, then summarize with LLMs
- **📓 Obsidian Vaults** - Browse, filter by tag, and summarize your markdown notes
- **🌐 Web Pages** - Fetch and summarize any web content

Supports multiple LLM providers: **Claude**, **OpenAI**, **Gemini**, **DeepSeek**

## Installation

```bash
# Clone the repo
git clone https://github.com/your-username/modelO_kit.git
cd modelO_kit

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Configuration

Create a `.env` file with your API keys:

```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
DEEPSEEK_API_KEY=...
```

## Usage

### CLI

```bash
uv run modelo-kit
```

```
🧠 Modelo Kit - Content Parser & Summarizer
==================================================

Select content source:
  1. 📄 Research Paper (PDF file or URL)
  2. 📓 Obsidian Vault (markdown notes)
  3. 🌐 Web Page (URL)
  4. ❌ Exit
```

### Programmatic

```python
from modelo_kit.parsers import PaperParser
from modelo_kit.services.obsidian_vault import ObsidianVault
from modelo_kit.llm_factory import get_llm_service

# Parse a research paper
parser = PaperParser()
paper = parser.parse("https://arxiv.org/pdf/1234.5678.pdf", load_content=True)
parser.print_structure(paper)

# Browse an Obsidian vault
vault = ObsidianVault("~/Documents/MyVault")
for note in vault.notes_with_tag("machine-learning"):
    print(f"{note.title}: {note.word_count} words")

# Summarize with an LLM
llm = get_llm_service("claude", "claude-sonnet-4-5-20250929")
llm.set_system_prompt("You are a helpful assistant.")
summary = llm.ask(f"Summarize: {paper.abstract}")
```

## Project Structure

```
src/modelo_kit/
├── main.py                 # CLI entry point
├── llm_factory.py          # LLM provider factory
├── prompts.yaml            # Prompt templates
├── models/                 # Pydantic data models
├── parsers/                # PDF and markdown parsers
├── llms/                   # LLM service implementations
└── services/
    ├── obsidian_vault.py   # Obsidian vault management
    ├── prompt_service.py   # YAML prompt loading
    ├── summarizer_service.py
    └── web_service.py
```

## Development

```bash
# Run linting
uv run ruff check src/ --fix
uv run ruff format src/

# Type checking
uv run mypy src/

# Run tests
uv run pytest
```

## How It Works

### Paper Parsing

Uses a **facade pattern** to auto-select between TOC-based and font-based PDF parsing strategies. Handles both local files and URLs transparently.

### Obsidian Integration

Parses markdown notes including:
- YAML frontmatter
- `[[wiki links]]` and `![[embeds]]`
- `#tags` (inline and frontmatter)
- Backlink resolution

### LLM Abstraction

All LLM providers implement `BaseLLMService` with:
- `ask(prompt)` - Simple text completion
- `ask_structured(prompt, schema)` - Structured output with Pydantic
- `set_system_prompt(prompt)` - Configure system behavior

## License

MIT