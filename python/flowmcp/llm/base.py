from abc import ABC, abstractmethod
import asyncio


class LLMBase(ABC):
    last_usage: dict = {}

    @abstractmethod
    def complete(self, prompt: str) -> str:
        pass

    async def acomplete(self, prompt: str) -> str:
        return await asyncio.to_thread(self.complete, prompt)


class LLM:
    """
    Factory for creating LLM clients from a simple string.

    Supported formats:
        "openai/gpt-4"
        "openai-compat:local-model"
        "anthropic/claude-3-5-sonnet"
        "ollama/llama3"
        "gemini/gemini-pro"
    """

    @staticmethod
    def from_string(model_string: str, **kwargs) -> LLMBase:
        if model_string.startswith("openai-compat:"):
            provider = "openai-compat"
            model = model_string.split(":", 1)[1].strip()
        elif "/" in model_string:
            provider, model = model_string.split("/", 1)
            model = model.strip()
        else:
            raise ValueError(
                f"Invalid model format '{model_string}'. "
                "Use 'provider/model' e.g. 'openai/gpt-4' or 'openai-compat:<model>'"
            )

        if not model:
            raise ValueError(f"Missing model name in '{model_string}'.")

        if provider == "openai-compat":
            unexpected = set(kwargs) - {"base_url", "api_key"}
            if unexpected:
                bad = ", ".join(sorted(unexpected))
                raise ValueError(
                    f"Unexpected keyword argument(s) for 'openai-compat': {bad}"
                )
            from .openai_compat import OpenAICompatClient

            return OpenAICompatClient(
                model,
                base_url=kwargs.get("base_url"),
                api_key=kwargs.get("api_key"),
            )

        if kwargs:
            bad = ", ".join(sorted(kwargs))
            raise ValueError(
                f"Unexpected keyword argument(s) for provider '{provider}': {bad}"
            )

        if provider == "openai":
            from .openai import OpenAIClient

            return OpenAIClient(model)
        elif provider == "anthropic":
            from .anthropic import AnthropicClient

            return AnthropicClient(model)
        elif provider == "ollama":
            from .ollama import OllamaClient

            return OllamaClient(model)
        elif provider == "gemini":
            from .gemini import GeminiClient

            return GeminiClient(model)
        elif provider == "mistral":
            from .mistral import MistralClient

            return MistralClient(model)
        elif provider == "groq":
            from .groq import GroqClient

            return GroqClient(model)
        elif provider == "huggingface":
            from .huggingface import HuggingFaceClient

            return HuggingFaceClient(model)
        else:
            raise ValueError(
                f"Unsupported provider '{provider}'. Supported: "
                "openai, openai-compat, anthropic, ollama, gemini, "
                "mistral, groq, huggingface"
            )
