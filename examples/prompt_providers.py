"""
Example: Using modelO_kit with floating_prompts database.

This demonstrates how to configure modelO_kit to read prompts from
the PostgreSQL database instead of the default YAML file.

Setup:
    1. Ensure floating_prompts database is running (docker-compose up -d)
    2. Run migrations (uv run alembic upgrade head)
    3. Install with db support (uv sync --extra db)
    4. Set environment variables (.env file)
"""

from modelo_kit.services import (
    LLMRequest,
    LLMService,
    PromptService,
    # Providers for advanced usage
    DatabasePromptProvider,
    YamlPromptProvider,
    CachedPromptProvider,
)


def example_yaml_prompts() -> None:
    """
    Example 1: Default behavior - using YAML prompts.
    
    This is the default and requires no additional setup.
    """
    print("=" * 60)
    print("Example 1: YAML Prompts (Default)")
    print("=" * 60)
    
    # Default singleton - uses prompts.yaml
    service = PromptService()
    
    # Or explicitly create with YAML provider
    service = PromptService.from_yaml()
    
    # Get a prompt
    prompt = service.get(
        "paper_section_summary",
        section_title="Introduction",
        section_content="This paper explores...",
    )
    
    print(f"System: {prompt.system_prompt[:50]}...")
    print(f"User: {prompt.user_prompt[:50]}...")


def example_database_prompts() -> None:
    """
    Example 2: Using database prompts from floating_prompts.
    
    Requires:
    - floating_prompts package installed
    - PostgreSQL database running
    - Environment variables configured
    """
    print("\n" + "=" * 60)
    print("Example 2: Database Prompts")
    print("=" * 60)
    
    try:
        # Create service with database provider
        service = PromptService.from_database()
        
        # Get latest version of a prompt
        prompt = service.get("summarizer", content="Hello world!")
        print(f"Latest: {prompt.user_prompt}")
        
        # Get specific version
        service_v1 = PromptService.from_database(version=1)
        prompt_v1 = service_v1.get("summarizer", content="Hello world!")
        print(f"Version 1: {prompt_v1.user_prompt}")
        
    except ImportError:
        print("floating_prompts not installed. Install with: uv sync --extra db")
    except Exception as e:
        print(f"Database error: {e}")


def example_cached_provider() -> None:
    """
    Example 3: Using cached provider for performance.
    
    The CachedPromptProvider wraps any provider to add caching,
    reducing database calls in high-throughput scenarios.
    """
    print("\n" + "=" * 60)
    print("Example 3: Cached Provider")
    print("=" * 60)
    
    try:
        # Create a cached database provider
        db_provider = DatabasePromptProvider()
        cached_provider = CachedPromptProvider(db_provider, ttl_seconds=300)
        
        # Use with PromptService
        service = PromptService(provider=cached_provider)
        
        # First call - hits database
        prompt1 = service.get("summarizer", content="First call")
        print("First call: hit database")
        
        # Second call - from cache
        prompt2 = service.get("summarizer", content="First call")
        print("Second call: from cache")
        
        # Clear cache if needed
        cached_provider.clear_cache()
        print("Cache cleared")
        
    except ImportError:
        print("floating_prompts not installed")


def example_with_llm_service() -> None:
    """
    Example 4: Full integration with LLMService.
    
    Shows how to use database prompts with the LLMService
    for actual LLM calls.
    """
    print("\n" + "=" * 60)
    print("Example 4: Full LLM Integration")
    print("=" * 60)
    
    try:
        # Option A: Use database prompts with LLMService
        db_prompts = PromptService.from_database()
        llm_service = LLMService(prompts=db_prompts)
        llm_service.configure("claude", "claude-sonnet-4-5-20250929")
        
        # Make a request using a database prompt
        response = llm_service.generate(
            LLMRequest(
                prompt_key="summarizer",
                variables={"content": "This is a test document about AI."},
            )
        )
        print(f"Response: {response[:100]}...")
        
    except ImportError:
        print("floating_prompts not installed")
    except Exception as e:
        print(f"Error: {e}")


def example_fallback_pattern() -> None:
    """
    Example 5: Fallback pattern - try database, fallback to YAML.
    
    A resilient pattern for production where you want database
    prompts but fall back to YAML if the database is unavailable.
    """
    print("\n" + "=" * 60)
    print("Example 5: Fallback Pattern")
    print("=" * 60)
    
    def get_prompt_service() -> PromptService:
        """Get prompt service with database fallback to YAML."""
        try:
            # Try database first
            service = PromptService.from_database()
            # Test connection by listing keys
            _ = service.list_keys()
            print("Using database prompts")
            return service
        except Exception as e:
            print(f"Database unavailable ({e}), falling back to YAML")
            return PromptService.from_yaml()
    
    service = get_prompt_service()
    print(f"Available prompts: {service.list_keys()}")


def example_list_and_check() -> None:
    """
    Example 6: List and check prompts.
    """
    print("\n" + "=" * 60)
    print("Example 6: List and Check Prompts")
    print("=" * 60)
    
    service = PromptService()  # Default YAML
    
    # List all available prompts
    keys = service.list_keys()
    print(f"Available prompts: {keys}")
    
    # Check if a prompt exists
    if service.exists("paper_section_summary"):
        print("'paper_section_summary' exists")
    
    if not service.exists("nonexistent_prompt"):
        print("'nonexistent_prompt' does not exist")


if __name__ == "__main__":
    # Run all examples
    example_yaml_prompts()
    example_list_and_check()
    
    # These require floating_prompts to be installed and configured
    # example_database_prompts()
    # example_cached_provider()
    # example_with_llm_service()
    # example_fallback_pattern()
