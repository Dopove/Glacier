from typing import List, Dict, Any, Union
import orjson
from .base import BaseProvider

class AnthropicAdapter(BaseProvider):
    """
    Adapter for Anthropic's Messages API format.
    """
    def map_multimodal_content(self, content: Any) -> Union[str, List[Dict[str, Any]]]:
        if isinstance(content, str): 
            return content
        
        mapped_content = []
        for block in content:
            block_dict = block if isinstance(block, dict) else block.model_dump()
            block_type = block_dict.get('type')
            
            if block_type == "text":
                mapped_content.append({"type": "text", "text": block_dict.get('text')})
            elif block_type == "image":
                source = block_dict.get("source", {})
                if source.get("type") == "base64":
                    mapped_content.append({
                        "type": "image_url", 
                        "image_url": {"url": f"data:{source.get('media_type')};base64,{source.get('data')}"}
                    })
            # The original implementation mapped tool calls/results to text. This is a robust way to handle them.
            elif block_type == "tool_use":
                mapped_content.append({
                    "type": "text", 
                    "text": f"<tool_call id='{block_dict.get('id')}'>{block_dict.get('name')}({orjson.dumps(block_dict.get('input')).decode()})</tool_call>"
                })
            elif block_type == "tool_result":
                 mapped_content.append({
                    "type": "text", 
                    "text": f"<tool_result tool_use_id='{block_dict.get('tool_use_id')}'>{block_dict.get('content')}</tool_result>"
                })
                
        return mapped_content
