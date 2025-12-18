from modelo_kit.llm_factory import get_llm_service
from modelo_kit.services.prompt_service import PromptService
from modelo_kit.services.web_service import WebService

def main():
    providers = {
        "openai": "gpt-5",
        "deepseek": "deepseek-chat",
        "claude": "claude-opus-4-5-20251101",
        "gemini": "gemini-3-pro-preview",
    }

    url = input("Enter a website URL to fetch content from: ").strip()
    print(f"\nFetching content from: {url}\n")
    content = WebService.get_web_content(url)
    if not content:
        print("No content could be fetched from the provided URL.")
        return

    print("\n--- Webpage Content ---\n")
    print(content)

    prompt_service = PromptService()
    # Use the appropriate prompt key and pass content as a variable
    prompt_obj = prompt_service.get("webpage_summary", content=content)

    for provider, model_name in providers.items():
        print(f"\n--- {provider.upper()} ({model_name}) ---")
        try:
            llm = get_llm_service(provider, model_name)
            if prompt_obj.system_prompt:
                llm.set_system_prompt(prompt_obj.system_prompt)
                llm_input = prompt_obj.user_prompt
            else:
                # Only user prompt present
                llm_input = prompt_obj.user_prompt

            answer = llm.ask(llm_input)
            print(answer)
        except Exception as e:
            print(f"Error with {provider}: {e}")

if __name__ == "__main__":
    main()