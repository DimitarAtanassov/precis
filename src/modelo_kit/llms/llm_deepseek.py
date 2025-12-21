import os
import time
from typing import TypeVar

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

from modelo_kit.llms.llm_base import BaseLLMService, StructuredOutputError

T = TypeVar("T", bound=BaseModel)


class DeepSeekLLMService(BaseLLMService):
    """DeepSeek LLM service using OpenAI-compatible API."""

    API_BASE = "https://api.deepseek.com/v1"
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0

    def __init__(self, model_name: str = "deepseek-chat") -> None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment.")

        self.chat = ChatOpenAI(
            api_key=SecretStr(api_key), model=model_name, base_url=self.API_BASE
        )
        self.model_name = model_name
        self.system_prompt: str | None = None

    def set_system_prompt(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt

    def ask(self, prompt: str) -> str:
        messages: list[BaseMessage] = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        messages.append(HumanMessage(content=prompt))
        response = self.chat.invoke(messages)
        content = response.content
        if not isinstance(content, str):
            content = str(content)
        return content

    def ask_structured(
        self, prompt: str, output_schema: type[T], system_prompt: str | None = None
    ) -> T:
        messages: list[BaseMessage] = []
        sys_prompt = system_prompt or self.system_prompt
        if sys_prompt:
            messages.append(SystemMessage(content=sys_prompt))
        messages.append(HumanMessage(content=prompt))

        last_error = None

        for attempt in range(self.MAX_RETRIES):
            try:
                structured_llm = self.chat.with_structured_output(output_schema)
                result = structured_llm.invoke(messages)

                if result is None:
                    raise StructuredOutputError(
                        f"DeepSeek returned None for {output_schema.__name__}"
                    )
                return result  # type: ignore[return-value]

            except Exception as e:
                last_error = e
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
                    continue

        if last_error is None:
            last_error = Exception("Unknown error")
        return self._fallback_parse(messages, output_schema, last_error)

    def _fallback_parse(
        self,
        messages: list[BaseMessage],
        output_schema: type[T],
        original_error: Exception,
    ) -> T:
        """Fallback: get raw response and manually construct the schema."""
        try:
            response = self.chat.invoke(messages)
            content = response.content

            schema_name = output_schema.__name__

            if schema_name == "SectionSummaryOutput":
                return output_schema(
                    summary=(
                        content[:500]
                        if content
                        else "Unable to generate summary."
                    ),
                    key_points=[],
                )
            elif schema_name == "ExecutiveSummaryOutput":
                return output_schema(
                    executive_summary=content
                    if content
                    else "Unable to generate executive summary."
                )
            elif schema_name == "ContributionsOutput":
                if isinstance(content, str):
                    lines = [
                        line.strip()
                        for line in content.split("\n")
                        if isinstance(line, str) and line.strip()
                    ]
                    contributions = [
                        line.lstrip("0123456789.-) ") for line in lines[:7]
                    ]
                else:
                    contributions = []
                return output_schema(
                    contributions=contributions or ["Unable to extract contributions."]
                )
            elif schema_name == "ChunkSummaryOutput":
                return output_schema(
                    updated_summary=content
                    if content
                    else "Unable to generate chunk summary."
                )
            else:
                raise StructuredOutputError(
                    f"Failed to parse {schema_name}: {original_error}"
                )

        except Exception as fallback_error:
            raise StructuredOutputError(
                f"Structured output failed for {output_schema.__name__}. "
                f"Original error: {original_error}. "
                f"Fallback error: {fallback_error}"
            ) from fallback_error
