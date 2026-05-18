from .base import BaseProvider
from .anthropic import AnthropicAdapter
from .google import GeminiAdapter
from .openai import OpenAIAdapter

def get_provider_adapter(model_name: str = None, provider_hint: str = None) -> BaseProvider:
    """
    Factory function to return the correct Provider Adapter.
    """
    model_name_lower = (model_name or "").lower()
    
    if provider_hint == "anthropic" or any(k in model_name_lower for k in ["claude", "anthropic"]):
        return AnthropicAdapter()
    elif provider_hint == "google" or any(k in model_name_lower for k in ["gemini", "google"]):
        return GeminiAdapter()
    else:
        # Default to OpenAI format for OpenAI models, Ollama, and generic endpoints
        return OpenAIAdapter()