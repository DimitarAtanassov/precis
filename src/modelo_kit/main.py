"""
Parse and analyze a research paper.
"""

import sys
from pathlib import Path

from modelo_kit.llm_factory import get_llm_service
from modelo_kit.models.paper_model import ParsedPaper, Section
from modelo_kit.models.summary_model import PaperSummary
from modelo_kit.parsers import PaperParser
from modelo_kit.services.summarizer_service import PaperSummarizer, SummarizerConfig


def main() -> None:
    print("\n📚 Research Paper Parser & Summarizer")
    print("=" * 40)
    path_or_url = _get_input_path()
    if not path_or_url:
        print("❌ No input provided.")
        sys.exit(1)

    paper = _parse_paper(path_or_url)
    _print_structure(paper)
    print("\n" + "=" * 40)
    if not _confirm("Generate LLM summary? (y/n): "):
        _save_and_report_parsed(paper, path_or_url)
        return

    provider, model = _select_llm()
    llm = get_llm_service(provider, model)
    config = SummarizerConfig.for_model(model, verbose=True)
    summarizer = PaperSummarizer(llm, config)
    _print_summary_info(config, model)
    summary = summarizer.summarize(paper)
    _save_summary(summary, path_or_url)
    _print_executive_summary(summary)
    _save_and_report_parsed(paper, path_or_url)


def _get_input_path() -> str:
    """Prompt user for PDF path or URL."""
    return input("Enter PDF path or URL: ").strip()


def _parse_paper(path_or_url: str) -> ParsedPaper:
    """Parse the paper using PaperParser."""
    parser = PaperParser()
    print("\n🔍 Parsing paper...")
    return parser.parse(path_or_url, load_content=True)


def _print_structure(paper: ParsedPaper) -> None:
    """Print the structure of the parsed paper."""
    parser = PaperParser()
    parser.print_structure(paper)


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


def _save_summary(summary: PaperSummary, path_or_url: str) -> None:
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


def _save_and_report_parsed(paper: ParsedPaper, path_or_url: str) -> None:
    """Save parsed content and report location."""
    output_name = _get_output_name(path_or_url, "_parsed.txt")
    _save_parsed_content(paper, output_name)
    print(f"\n✅ Parsed content saved to: {output_name}")


def _get_output_name(path_or_url: str, suffix: str) -> str:
    """Generate output filename from input."""
    if path_or_url.startswith(("http://", "https://")):
        paper_id = path_or_url.split("/")[-1].replace(".pdf", "")
        return f"{paper_id}{suffix}"
    return Path(path_or_url).stem + suffix


def _save_parsed_content(paper: ParsedPaper, output_path: str) -> None:
    """Save parsed paper content to file."""
    with open(output_path, "w", encoding="utf-8") as f:
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


if __name__ == "__main__":
    main()
