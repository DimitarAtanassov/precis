import os
import time
from typing import Optional, Type, TypeVar
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from modelo_kit.llms.llm_base import BaseLLMService, StructuredOutputError

T = TypeVar("T", bound=BaseModel)


class OpenAILLMService(BaseLLMService):
    """OpenAI LLM service with structured output support."""
    
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0

    def __init__(self, model_name="gpt-4o"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment.")
        self.chat = ChatOpenAI(api_key=api_key, model=model_name)
        self.model_name = model_name
        self.system_prompt = None

    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt

    def ask(self, prompt: str) -> str:
        messages = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        messages.append(HumanMessage(content=prompt))
        response = self.chat.invoke(messages)
        return response.content

    def ask_structured(
        self, 
        prompt: str, 
        output_schema: Type[T],
        system_prompt: Optional[str] = None
    ) -> T:
        """Get structured output with retry logic."""
        messages = []
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
                        f"OpenAI returned None for {output_schema.__name__}"
                    )
                
                return result
                
            except Exception as e:
                last_error = e
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
                    continue
        
        return self._fallback_parse(messages, output_schema, last_error)

    def _fallback_parse(
        self, 
        messages: list, 
        output_schema: Type[T],
        original_error: Exception
    ) -> T:
        """Fallback: get raw response and manually construct the schema."""
        try:
            response = self.chat.invoke(messages)
            content = response.content
            
            schema_name = output_schema.__name__
            
            if schema_name == "SectionSummaryOutput":
                return output_schema(
                    summary=content[:500] if content else "Unable to generate summary.",
                    key_points=[]
                )
            elif schema_name == "ExecutiveSummaryOutput":
                return output_schema(
                    executive_summary=content if content else "Unable to generate executive summary."
                )
            elif schema_name == "ContributionsOutput":
                lines = [l.strip() for l in content.split("\n") if l.strip()]
                contributions = [l.lstrip("0123456789.-) ") for l in lines[:7]]
                return output_schema(contributions=contributions or ["Unable to extract contributions."])
            elif schema_name == "ChunkSummaryOutput":
                return output_schema(
                    updated_summary=content if content else "Unable to generate chunk summary."
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
            )