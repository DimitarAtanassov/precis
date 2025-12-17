from modelo_kit.llm_factory import get_llm_service
from modelo_kit.services.prompt_service import PromptService

def main():
    providers = {
        "openai": "gpt-5",
        "deepseek": "deepseek-chat",
        "claude": "claude-opus-4-5-20251101",
        "gemini": "gemini-3-pro-preview",
    }
    # Use PromptService to get the question prompt with dynamic variable
    prompt_service = PromptService()
    prompt = prompt_service.get("question", country="France")

    for provider, model_name in providers.items():
        print(f"\n--- {provider.upper()} ({model_name}) ---")
        try:
            llm = get_llm_service(provider, model_name)
            answer = llm.ask(prompt)
            print(answer)
        except Exception as e:
            print(f"Error with {provider}: {e}")

if __name__ == "__main__":
    main()