import os
from typing import Optional
from .base import LLMBase


class MistralClient(LLMBase):
    """
    Mistral AI LLM client using the mistralai Python SDK.

    Usage:
        llm = LLM.from_string("mistral/mistral-small-latest")
    """

    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")

    def complete(self, prompt: str) -> str:
        try:
            from mistralai import Mistral
        except ImportError:
            raise ImportError("Install mistralai: pip install mistralai")
        client = Mistral(api_key=self.api_key)
        response = client.chat.complete(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
