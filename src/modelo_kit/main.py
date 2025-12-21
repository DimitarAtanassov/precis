"""
Parse and analyze a research paper.
"""
import sys
from pathlib import Path

from modelo_kit.parsers import PaperParser
from modelo_kit.services.summarizer_service import PaperSummarizer, SummarizerConfig
from modelo_kit.llm_factory import get_llm_service


def main():
    print("\n📚 Research Paper Parser & Summarizer")
    print("=" * 40)
    path_or_url = input("Enter PDF path or URL: ").strip()
    
    if not path_or_url:
        print("❌ No input provided.")
        sys.exit(1)
    
    # Parse the paper
    parser = PaperParser()
    print("\n🔍 Parsing paper...")
    paper = parser.parse(path_or_url, load_content=True)
    parser.print_structure(paper)
    
    # Ask if user wants a summary
    print("\n" + "=" * 40)
    summarize = input("Generate LLM summary? (y/n): ").strip().lower()
    
    if summarize == "y":
        # Initialize LLM
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
        
        # Let user customize model name
        print(f"\nDefault model: {default_model}")
        custom_model = input("Enter model name (or press Enter for default): ").strip()
        model = custom_model if custom_model else default_model
        
        llm = get_llm_service(provider, model)
        
        # Use model-aware config (auto-detects token limits)
        config = SummarizerConfig.for_model(model, verbose=True)
        summarizer = PaperSummarizer(llm, config)
        
        print(f"\n🤖 Generating summary with {model}...")
        print(f"   Context window: {config.max_chunk_tokens * 3:,} tokens")
        print(f"   Max chunk size: {config.max_chunk_tokens:,} tokens")
        summary = summarizer.summarize(paper)
        
        # Save summary
        output_name = _get_output_name(path_or_url, "_summary.md")
        with open(output_name, "w") as f:
            f.write(summary.to_markdown())
        
        print(f"\n✅ Summary saved to: {output_name}")
        
        # Print executive summary
        print("\n" + "=" * 60)
        print("EXECUTIVE SUMMARY")
        print("=" * 60)
        print(summary.executive_summary)
        
        if summary.key_contributions:
            print("\nKEY CONTRIBUTIONS:")
            for i, c in enumerate(summary.key_contributions, 1):
                print(f"  {i}. {c}")
    
    # Also save parsed content
    output_name = _get_output_name(path_or_url, "_parsed.txt")
    _save_parsed_content(paper, output_name)
    print(f"\n✅ Parsed content saved to: {output_name}")


def _get_output_name(path_or_url: str, suffix: str) -> str:
    """Generate output filename from input."""
    if path_or_url.startswith(("http://", "https://")):
        paper_id = path_or_url.split("/")[-1].replace(".pdf", "")
        return f"{paper_id}{suffix}"
    return Path(path_or_url).stem + suffix


def _save_parsed_content(paper, output_path: str):
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
        
        def write_sections(sections, indent=0):
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