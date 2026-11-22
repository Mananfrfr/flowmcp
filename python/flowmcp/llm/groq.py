import os
from typing import Optional
from .base import LLMBase


class GroqClient(LLMBase):
    """
    Groq LLM client for fast inference using the groq Python SDK.

    Usage:
        llm = LLM.from_string("groq/llama3-8b-8192")
    """

    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")

    def complete(self, prompt: str) -> str:
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("Install groq: pip install groq")
        client = Groq(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
