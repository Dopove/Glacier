from typing import List, Dict, Any, Union
import orjson
from .base import BaseProvider

class GeminiAdapter(BaseProvider):
    """
    Adapter for Google's Gemini API format.
    """
    def map_multimodal_content(self, parts: List[Any]) -> Union[str, List[Dict[str, Any]]]:
        if isinstance(parts, str): 
            return parts # Should not happen with Gemini, but defensive
            
        mapped_content = []
        for part in parts:
            part_dict = part if isinstance(part, dict) else part.model_dump(exclude_none=True)
            
            if "text" in part_dict:
                mapped_content.append({"type": "text", "text": part_dict["text"]})
            elif "inline_data" in part_dict:
                data = part_dict["inline_data"].get("data")
                mime_type = part_dict["inline_data"].get("mime_type")
                if data and mime_type:
                    mapped_content.append({
                        "type": "image_url", 
                        "image_url": {"url": f"data:{mime_type};base64,{data}"}
                    })
            # The original implementation mapped tool calls/results to text.
            elif "functionCall" in part_dict:
                mapped_content.append({
                    "type": "text", 
                    "text": f"<functionCall>{orjson.dumps(part_dict['functionCall']).decode()}</functionCall>"
                })
            elif "functionResponse" in part_dict:
                mapped_content.append({
                    "type": "text", 
                    "text": f"<functionResponse>{orjson.dumps(part_dict['functionResponse']).decode()}</functionResponse>"
                })
        return mapped_content
