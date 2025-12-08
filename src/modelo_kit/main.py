from modelo_kit.llm_factory import get_llm_service

def main():
    provider = "openai"  # or "deepseek", "claude", "gemini"
    model_name = "gpt-3.5-turbo"  # or other model names
    llm = get_llm_service(provider, model_name)
    answer = llm.ask("Hello, LLM!")
    print(answer)

if __name__ == "__main__":
    main()