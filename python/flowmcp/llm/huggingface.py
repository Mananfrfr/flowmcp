import os
from typing import Optional
from .base import LLMBase


class HuggingFaceClient(LLMBase):
    """
    HuggingFace Inference API client for accessing hosted open-source models.

    Usage:
        llm = LLM.from_string("huggingface/mistralai/Mistral-7B-Instruct-v0.3")
    """

    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("HF_TOKEN")

    def complete(self, prompt: str) -> str:
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise ImportError("Install huggingface_hub: pip install huggingface_hub")
        client = InferenceClient(model=self.model, token=self.api_key)
        response = client.text_generation(prompt, max_new_tokens=512)
        return response
