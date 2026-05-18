from typing import List, Dict, Any, Union
from .base import BaseProvider

class OpenAIAdapter(BaseProvider):
    """
    Adapter for OpenAI and compatible endpoints (like Ollama).
    The content is already in the target format, so this is a pass-through.
    """
    def map_multimodal_content(self, content: Any) -> Union[str, List[Dict[str, Any]]]:
        return content
