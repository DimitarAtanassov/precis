"""
Modelo Kit - Parse and summarize content from various sources.

Supports:
- Research papers (PDF files and URLs)
- Obsidian markdown vaults
- Web pages
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

from modelo_kit.llm_factory import get_llm_service
from modelo_kit.models import ObsidianNote, Paper, PaperSummary, Section
from modelo_kit.parsers import PaperParser
from modelo_kit.services.obsidian_vault import ObsidianVault
from modelo_kit.services.prompt_service import PromptService
from modelo_kit.services.summarizer_service import PaperSummarizer, SummarizerConfig
from modelo_kit.services.web_service import WebService

load_dotenv()

# Initialize prompt service (singleton)
_prompts = PromptService()


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Main entry point for the CLI."""
    print("\n🧠 Modelo Kit - Content Parser & Summarizer")
    print("=" * 50)

    mode = _select_mode()

    if mode == "1":
        _handle_paper_mode()
    elif mode == "2":
        _handle_obsidian_mode()
    elif mode == "3":
        _handle_web_mode()
    else:
        print("👋 Goodbye!")
        sys.exit(0)


def _select_mode() -> str:
    """Prompt user to select content source."""
    print("\nSelect content source:")
    print("  1. 📄 Research Paper (PDF file or URL)")
    print("  2. 📓 Obsidian Vault (markdown notes)")
    print("  3. 🌐 Web Page (URL)")
    print("  4. ❌ Exit")
    return input("\nChoice (1-4): ").strip()


# =============================================================================
# Paper Mode
# =============================================================================


def _handle_paper_mode() -> None:
    """Handle research paper parsing and summarization."""
    print("\n📄 Research Paper Mode")
    print("-" * 40)

    path_or_url = input("Enter PDF path or URL: ").strip()
    if not path_or_url:
        print("❌ No input provided.")
        return

    paper = _parse_paper(path_or_url)
    _print_paper_structure(paper)

    if not _confirm("\nGenerate LLM summary? (y/n): "):
        _save_parsed_paper(paper, path_or_url)
        return

    provider, model = _select_llm()
    llm = get_llm_service(provider, model)
    config = SummarizerConfig.for_model(model, verbose=True)
    summarizer = PaperSummarizer(llm, config)

    _print_summary_info(config, model)
    summary = summarizer.summarize(paper)

    _save_paper_summary(summary, path_or_url)
    _print_executive_summary(summary)
    _save_parsed_paper(paper, path_or_url)


def _parse_paper(path_or_url: str) -> Paper:
    """Parse the paper using PaperParser."""
    parser = PaperParser()
    print("\n🔍 Parsing paper...")
    return parser.parse(path_or_url, load_content=True)


def _print_paper_structure(paper: Paper) -> None:
    """Print the structure of the parsed paper."""
    parser = PaperParser()
    parser.print_structure(paper)


def _save_parsed_paper(paper: Paper, path_or_url: str) -> None:
    """Save parsed paper content to file."""
    output_name = _get_output_name(path_or_url, "_parsed.txt")

    with open(output_name, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"TITLE: {paper.title}\n")
        f.write(f"SOURCE: {paper.source_path}\n")
        f.write(f"PAGES: {paper.total_pages}\n")
        f.write("=" * 80 + "\n\n")

        if paper.abstract:
            f.write("ABSTRACT\n")
            f.write("-" * 40 + "\n")
            f.write(paper.abstract + "\n\n")

        def write_sections(sections: list[Section], indent: int = 0) -> None:
            for section in sections:
                prefix = "  " * indent
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"{prefix}SECTION: {section.title}\n")
                f.write(f"{prefix}Pages: {section.page_start} - {section.page_end}\n")
                f.write("-" * 80 + "\n\n")
                if section.content:
                    f.write(section.content + "\n")
                if section.subsections:
                    write_sections(section.subsections, indent + 1)

        write_sections(paper.sections)

    print(f"\n✅ Parsed content saved to: {output_name}")


def _save_paper_summary(summary: PaperSummary, path_or_url: str) -> None:
    """Save summary to markdown file."""
    output_name = _get_output_name(path_or_url, "_summary.md")
    with open(output_name, "w", encoding="utf-8") as f:
        f.write(summary.to_markdown())
    print(f"\n✅ Summary saved to: {output_name}")


def _print_executive_summary(summary: PaperSummary) -> None:
    """Print executive summary and key contributions."""
    print("\n" + "=" * 60)
    print("EXECUTIVE SUMMARY")
    print("=" * 60)
    print(summary.executive_summary)
    if summary.key_contributions:
        print("\nKEY CONTRIBUTIONS:")
        for i, c in enumerate(summary.key_contributions, 1):
            print(f"  {i}. {c}")


# =============================================================================
# Obsidian Mode
# =============================================================================


def _handle_obsidian_mode() -> None:
    """Handle Obsidian vault parsing and summarization."""
    print("\n📓 Obsidian Vault Mode")
    print("-" * 40)

    vault_path = input("Enter vault path: ").strip()
    if not vault_path:
        print("❌ No path provided.")
        return

    path = Path(vault_path).expanduser()
    if not path.exists():
        print(f"❌ Vault not found: {path}")
        return

    vault = ObsidianVault(path)
    _print_vault_stats(vault)

    action = _select_obsidian_action()

    if action == "1":
        _summarize_single_note(vault)
    elif action == "2":
        _summarize_notes_by_tag(vault)
    elif action == "3":
        _summarize_all_notes(vault)
    elif action == "4":
        _list_vault_notes(vault)


def _select_obsidian_action() -> str:
    """Select action for Obsidian vault."""
    print("\nSelect action:")
    print("  1. 📝 Summarize a single note")
    print("  2. 🏷️  Summarize notes by tag")
    print("  3. 📚 Summarize all notes")
    print("  4. 📋 List all notes")
    print("  5. ↩️  Back to main menu")
    return input("\nChoice (1-5): ").strip()


def _print_vault_stats(vault: ObsidianVault) -> None:
    """Print vault statistics."""
    stats = vault.get_stats()
    print("\n📊 Vault Statistics:")
    print(f"   Notes: {stats.total_notes}")
    print(f"   Words: {stats.total_words:,}")
    print(f"   Links: {stats.total_links}")
    print(f"   Tags:  {len(stats.unique_tags)}")
    if stats.orphan_notes:
        print(f"   Orphans: {len(stats.orphan_notes)}")


def _summarize_single_note(vault: ObsidianVault) -> None:
    """Summarize a single note from the vault."""
    note_name = input("\nEnter note name (without .md): ").strip()
    if not note_name:
        print("❌ No note name provided.")
        return

    note = vault.get_note(note_name)
    if not note:
        print(f"❌ Note not found: {note_name}")
        return

    _print_note_info(note)

    if not _confirm("\nGenerate summary? (y/n): "):
        return

    provider, model = _select_llm()
    llm = get_llm_service(provider, model)

    # Get prompts from YAML
    prompt = _prompts.get(
        "obsidian_note_summary",
        title=note.title,
        tags=", ".join(note.tags) if note.tags else "none",
        word_count=str(note.word_count),
        content=note.body,
    )

    if prompt.system_prompt:
        llm.set_system_prompt(prompt.system_prompt)

    print(f"\n🤖 Summarizing with {model}...")
    response = llm.ask(prompt.user_prompt)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(response)

    _save_note_summary(note, response)


def _summarize_notes_by_tag(vault: ObsidianVault) -> None:
    """Summarize all notes with a specific tag."""
    stats = vault.get_stats()
    print(f"\nAvailable tags: {', '.join(stats.unique_tags[:20])}")
    if len(stats.unique_tags) > 20:
        print(f"   ... and {len(stats.unique_tags) - 20} more")

    tag = input("\nEnter tag to summarize: ").strip().lstrip("#")
    if not tag:
        print("❌ No tag provided.")
        return

    notes = list(vault.notes_with_tag(tag))
    if not notes:
        print(f"❌ No notes found with tag: #{tag}")
        return

    print(f"\n📝 Found {len(notes)} notes with #{tag}:")
    for note in notes[:10]:
        print(f"   • {note.title}")
    if len(notes) > 10:
        print(f"   ... and {len(notes) - 10} more")

    if not _confirm("\nGenerate combined summary? (y/n): "):
        return

    provider, model = _select_llm()
    llm = get_llm_service(provider, model)

    # Format notes content
    notes_content = "\n\n---\n\n".join(
        f"### {note.title}\n{note.body}" for note in notes
    )

    # Get prompts from YAML
    prompt = _prompts.get(
        "obsidian_notes_by_tag",
        tag=tag,
        notes_content=notes_content,
    )

    if prompt.system_prompt:
        llm.set_system_prompt(prompt.system_prompt)

    print(f"\n🤖 Summarizing {len(notes)} notes with {model}...")
    response = llm.ask(prompt.user_prompt)

    print("\n" + "=" * 60)
    print(f"SUMMARY: Notes tagged #{tag}")
    print("=" * 60)
    print(response)

    output_name = f"obsidian_tag_{tag}_summary.md"
    with open(output_name, "w", encoding="utf-8") as f:
        f.write(f"# Summary: Notes tagged #{tag}\n\n")
        f.write(f"*{len(notes)} notes summarized*\n\n")
        f.write(response)
    print(f"\n✅ Summary saved to: {output_name}")


def _summarize_all_notes(vault: ObsidianVault) -> None:
    """Summarize all notes in the vault."""
    notes = list(vault.notes())
    stats = vault.get_stats()

    print(f"\n⚠️  This will summarize {len(notes)} notes.")
    print("   This may take a while and use many API tokens.")

    if not _confirm("\nContinue? (y/n): "):
        return

    provider, model = _select_llm()
    llm = get_llm_service(provider, model)

    # Create notes overview
    notes_overview = "\n".join(
        f"- **{note.title}**: {note.word_count} words, "
        f"tags: {', '.join(note.tags) or 'none'}"
        for note in notes
    )

    # Get prompts from YAML
    prompt = _prompts.get(
        "obsidian_vault_overview",
        total_notes=str(stats.total_notes),
        total_words=str(stats.total_words),
        tag_count=str(len(stats.unique_tags)),
        notes_overview=notes_overview,
    )

    if prompt.system_prompt:
        llm.set_system_prompt(prompt.system_prompt)

    print(f"\n🤖 Creating vault overview with {model}...")
    response = llm.ask(prompt.user_prompt)

    print("\n" + "=" * 60)
    print("VAULT OVERVIEW")
    print("=" * 60)
    print(response)

    output_name = "obsidian_vault_summary.md"
    with open(output_name, "w", encoding="utf-8") as f:
        f.write("# Obsidian Vault Summary\n\n")
        f.write(f"*{len(notes)} notes analyzed*\n\n")
        f.write(response)
    print(f"\n✅ Summary saved to: {output_name}")


def _list_vault_notes(vault: ObsidianVault) -> None:
    """List all notes in the vault."""
    notes = list(vault.notes())
    print(f"\n📋 Notes in vault ({len(notes)} total):\n")

    for note in sorted(notes, key=lambda n: n.title.lower()):
        tags = f" [{', '.join(note.tags)}]" if note.tags else ""
        print(f"  • {note.title}{tags}")


def _print_note_info(note: ObsidianNote) -> None:
    """Print information about a note."""
    print(f"\n📝 Note: {note.title}")
    print(f"   Words: {note.word_count}")
    print(f"   Tags: {', '.join(note.tags) or 'none'}")
    print(f"   Links: {len(note.links)}")
    if note.backlinks:
        print(f"   Backlinks: {len(note.backlinks)}")


def _save_note_summary(note: ObsidianNote, summary: str) -> None:
    """Save a note summary to file."""
    output_name = f"{note.title}_summary.md"
    with open(output_name, "w", encoding="utf-8") as f:
        f.write(f"# Summary: {note.title}\n\n")
        f.write(summary)
    print(f"\n✅ Summary saved to: {output_name}")


# =============================================================================
# Web Mode
# =============================================================================


def _handle_web_mode() -> None:
    """Handle web page parsing and summarization."""
    print("\n🌐 Web Page Mode")
    print("-" * 40)

    url = input("Enter URL: ").strip()
    if not url:
        print("❌ No URL provided.")
        return

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    print(f"\n🔍 Fetching content from: {url}")
    content = WebService.get_web_content(url)

    if not content:
        print("❌ Could not fetch content from URL.")
        return

    word_count = len(content.split())
    print(f"✅ Fetched {word_count:,} words")

    # Show preview
    preview = content[:500] + "..." if len(content) > 500 else content
    print(f"\n📄 Preview:\n{preview}")

    if not _confirm("\nGenerate summary? (y/n): "):
        _save_web_content(url, content)
        return

    provider, model = _select_llm()
    llm = get_llm_service(provider, model)

    # Get prompts from YAML
    prompt = _prompts.get("webpage_summary", content=content)

    if prompt.system_prompt:
        llm.set_system_prompt(prompt.system_prompt)

    print(f"\n🤖 Summarizing with {model}...")
    response = llm.ask(prompt.user_prompt)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(response)

    _save_web_summary(url, response)
    _save_web_content(url, content)


def _save_web_content(url: str, content: str) -> None:
    """Save web content to file."""
    output_name = _get_output_name(url, "_content.txt")
    with open(output_name, "w", encoding="utf-8") as f:
        f.write(f"URL: {url}\n")
        f.write("=" * 80 + "\n\n")
        f.write(content)
    print(f"✅ Content saved to: {output_name}")


def _save_web_summary(url: str, summary: str) -> None:
    """Save web summary to file."""
    output_name = _get_output_name(url, "_summary.md")
    with open(output_name, "w", encoding="utf-8") as f:
        f.write(f"# Summary: {url}\n\n")
        f.write(summary)
    print(f"✅ Summary saved to: {output_name}")


# =============================================================================
# Shared Utilities
# =============================================================================


def _confirm(prompt: str) -> bool:
    """Prompt user for yes/no confirmation."""
    return input(prompt).strip().lower() == "y"


def _select_llm() -> tuple[str, str]:
    """Prompt user to select LLM provider and model."""
    print("\nSelect LLM provider:")
    print("  1. Claude")
    print("  2. OpenAI")
    print("  3. Gemini")
    print("  4. DeepSeek")
    choice = input("Choice (1-4): ").strip()

    providers = {
        "1": ("claude", "claude-sonnet-4-5-20250929"),
        "2": ("openai", "gpt-4o"),
        "3": ("gemini", "gemini-2.0-flash"),
        "4": ("deepseek", "deepseek-chat"),
    }

    if choice not in providers:
        print("Invalid choice, using Claude.")
        choice = "1"

    provider, default_model = providers[choice]
    print(f"\nDefault model: {default_model}")
    custom_model = input("Enter model name (or press Enter for default): ").strip()
    model = custom_model if custom_model else default_model

    return provider, model


def _print_summary_info(config: SummarizerConfig, model: str) -> None:
    """Print summary generation info."""
    print(f"\n🤖 Generating summary with {model}...")
    if config.max_chunk_tokens is not None:
        print(f"   Context window: {config.max_chunk_tokens * 3:,} tokens")
        print(f"   Max chunk size: {config.max_chunk_tokens:,} tokens")
    else:
        print("   Context window: unknown")
        print("   Max chunk size: unknown")


def _get_output_name(path_or_url: str, suffix: str) -> str:
    """Generate output filename from input."""
    if path_or_url.startswith(("http://", "https://")):
        # Extract meaningful name from URL
        name = path_or_url.split("/")[-1]
        name = name.replace(".pdf", "").replace(".html", "")
        # Handle URLs ending with /
        if not name:
            name = path_or_url.split("/")[-2]
        # Clean up query params
        name = name.split("?")[0]
        return f"{name}{suffix}"
    return Path(path_or_url).stem + suffix


if __name__ == "__main__":
    main()
