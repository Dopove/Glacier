from typing import List, Dict, Any, Union

class BaseProvider:
    """
    Abstract base class for provider-specific adapters.
    """
    def map_multimodal_content(self, content: Any) -> Union[str, List[Dict[str, Any]]]:
        """
        Maps a provider's specific multimodal content format into the standardized
        OpenAI-compatible format expected by the ICE Kernel's core processing logic.

        Args:
            content: The provider-specific content block (e.g., a string, a list of parts).

        Returns:
            A string or a list of standardized content blocks.
        """
        raise NotImplementedError("Each provider must implement its own content mapping logic.")
