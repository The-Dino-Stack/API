import os

from llm_providers import OpenAIProvider

def get_llm_provider():
    provider = os.getenv("LLM_PROVIDER")
    if not provider:
        raise ValueError("LLM provider is not set in environment variables.")

    api_key = os.getenv("LLM_PROVIDER_API_KEY")
    if not api_key:
        raise ValueError(f"API key for {provider} is not set in environment variables.")

    embedding_model = os.getenv("LLM_PROVIDER_EMBEDDING_MODEL")
    if not embedding_model:
        raise ValueError(f"Embedding model for {provider} is not set in environment variables.")

    llm_model = os.getenv("LLM_PROVIDER_LLM_MODEL")
    if not llm_model:
        raise ValueError(f"LLM model for {provider} is not set in environment variables.")

    if provider == "openai":
        return OpenAIProvider(api_key, embedding_model, llm_model)
    else:
        raise ValueError(f"Unknown provider: {provider}")