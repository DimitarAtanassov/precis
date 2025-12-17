import os
import yaml
from typing import Any, Dict
from pathlib import Path

class PromptService:
    _instance = None
    _prompts: Dict[str, str] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_prompts()
        return cls._instance

    def _load_prompts(self):
        # Find the root of the modelo_kit package
        package_root = Path(__file__).parent.parent
        prompts_path = package_root / "prompts.yaml"
        if not prompts_path.exists():
            raise FileNotFoundError(f"Could not find prompts.yaml at {prompts_path}")
        with open(prompts_path, "r") as f:
            self._prompts = yaml.safe_load(f)

    def get(self, key: str, **kwargs: Any) -> str:
        if key not in self._prompts:
            raise KeyError(f"Prompt '{key}' not found.")
        prompt = self._prompts[key]
        if kwargs:
            try:
                return prompt.format(**kwargs)
            except KeyError as e:
                raise ValueError(f"Missing variable for prompt '{key}': {e}")
        return prompt