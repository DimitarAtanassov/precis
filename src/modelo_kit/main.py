from modelo_kit.llm_service import LLMService

def main():
    llm = LLMService(model_name="gpt-5")
    answer = llm.ask("Hello, OpenAI!")
    print(answer)

if __name__ == "__main__":
    main()