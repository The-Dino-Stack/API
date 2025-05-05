from typing import List

import openai
from openai import OpenAI
from llm_providers import LLMProvider

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, embedding_model: str, llm_model: str):
        self.client = OpenAI(api_key=api_key)
        self.llm_model = llm_model
        self.embedding_model = embedding_model

    def validate(self):
        try:
            available_models = self.client.models.list()
            model_ids = [model.id for model in available_models.data]

            if self.llm_model not in model_ids:
                raise ValueError(f"❌ Model '{self.llm_model}' does not exist in your OpenAI account.")
        except openai.AuthenticationError as e:
            raise ValueError("❌ OpenAI API key is invalid. Aborting.") from e
        except Exception as e:
            raise RuntimeError(f"❌ Failed to validate OpenAI key: {e}") from e

    def embed_prompt(self, prompt: str) -> List[float]:
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=prompt
            )
            return response.data[0].embedding
        except Exception as e:
            raise RuntimeError(f"❌ Failed to generate embedding: {e}") from e

    def prompt(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"❌ Failed to generate response: {e}") from e