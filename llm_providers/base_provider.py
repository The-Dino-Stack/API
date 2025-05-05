from abc import ABC, abstractmethod
from typing import List, Dict

class LLMProvider(ABC):
    @abstractmethod
    def validate(self) -> None:
        """
        Validate credentials and model configuration.

        Raises:
            ValueError or RuntimeError if the setup is invalid (e.g. wrong API key, missing project)
        """
        pass

    @abstractmethod
    def embed_prompt(self, prompt: str) -> List[float]:
        """
        Generate an embedding for a given prompt.

        Args:
            prompt (str): The prompt to embed.

        Returns:
            List[float]: The embedding vector.
        """
        pass

    @abstractmethod
    def prompt(self, prompt: str) -> str:
        """
        Generate a response for a given prompt.

        Args:
            prompt (str): The prompt to send to the LLM.

        Returns:
            str: The response from the LLM.
        """
        pass