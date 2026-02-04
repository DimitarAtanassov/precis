from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from modelo_kit.models import Prompt


class PromptService:
    _instance: "PromptService | None" = None
    _prompts: dict[str, Any] = {}
    _loaded: bool = False

    def __new__(cls) -> "PromptService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load_prompts(self) -> None:
        """Load prompts from YAML file."""
        if self._loaded:
            return
        # Find the root of the modelo_kit package
        package_root = Path(__file__).parent.parent
        prompts_path = package_root / "prompts.yaml"
        if not prompts_path.exists():
            raise FileNotFoundError(f"Could not find prompts.yaml at {prompts_path}")
        with open(prompts_path) as f:
            self._prompts = yaml.safe_load(f)
        self._loaded = True

    def get(self, key: str, **kwargs: object) -> Prompt:
        """
        Returns a Prompt object with 'system_prompt' and 'user_prompt' attributes
        depending on the structure in prompts.yaml.

        If only a single prompt is present, returns Prompt(user_prompt=...).
        If both system and user prompts are present, returns both in a Prompt object.
        """
        # Ensure prompts are loaded
        self._load_prompts()

        if key not in self._prompts:
            raise KeyError(f"Prompt '{key}' not found.")

        prompt_entry = self._prompts[key]

        # Case 1: Both system and user prompts exist
        if (
            isinstance(prompt_entry, dict)
            and "system_prompt" in prompt_entry
            and "user_prompt" in prompt_entry
        ):
            system_prompt = prompt_entry["system_prompt"].get("prompt", "")
            user_prompt = prompt_entry["user_prompt"].get("prompt", "")
            # Format with kwargs if needed
            if kwargs:
                try:
                    user_prompt = user_prompt.format(**kwargs)
                    system_prompt = system_prompt.format(**kwargs)
                except KeyError as e:
                    raise ValueError(
                        f"Missing variable for prompt '{key}': {e}"
                    ) from None
            return Prompt(system_prompt=system_prompt, user_prompt=user_prompt)

        # Case 2: Only a single prompt
        if isinstance(prompt_entry, str):
            prompt = prompt_entry
        elif isinstance(prompt_entry, dict) and "prompt" in prompt_entry:
            prompt = prompt_entry["prompt"]
        else:
            raise ValueError(f"Prompt '{key}' has an unrecognized structure.")

        if kwargs:
            try:
                prompt = prompt.format(**kwargs)
            except KeyError as e:
                raise ValueError(f"Missing variable for prompt '{key}': {e}") from None

        return Prompt(user_prompt=prompt)
