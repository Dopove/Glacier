import logging
from typing import List, Optional, Dict, Any
import uuid
import datetime

from .local_persistence import LocalPersistence

logger = logging.getLogger("ice_reasoning")

class ReasoningManager:
    """
    Manages deep reasoning traces (Chain-of-Thought) for ICE-Lite threads.
    """

    def __init__(self, persistence_manager: LocalPersistence):
        self.persistence_manager = persistence_manager

    async def save_trace(
        self,
        tenant_id: str,
        session_id: str,
        content: str,
        thread_id: Optional[str] = "default",
        user_id: str = "default-user",
        parent_trace_id: Optional[str] = None,
        step_number: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ):
        """
        Saves a reasoning step to local storage.
        """
        traces = self.persistence_manager.load_reasoning_traces(tenant_id, user_id, session_id)
        
        trace_id = str(uuid.uuid4())
        created_at = datetime.datetime.now().isoformat()
        
        new_trace = {
            "id": trace_id,
            "tenant_id": tenant_id,
            "session_id": session_id,
            "thread_id": thread_id,
            "user_id": user_id,
            "parent_trace_id": parent_trace_id,
            "content": content,
            "embedding": embedding,
            "step_number": step_number,
            "metadata": metadata or {},
            "created_at": created_at
        }
        
        traces.append(new_trace)
        self.persistence_manager.save_reasoning_traces(tenant_id, user_id, session_id, traces)
        
        return trace_id

    async def get_active_chain(
        self, tenant_id: str, session_id: str, user_id: str = "default-user", limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the most recent reasoning steps for JIT context.
        """
        traces = self.persistence_manager.load_reasoning_traces(tenant_id, user_id, session_id)
        
        # Sort by created_at assuming chronological append, just to be safe
        # In a real DB this is ORDER BY created_at DESC LIMIT $4
        # We want the *last* 'limit' items
        if limit and len(traces) > limit:
            traces = traces[-limit:]
            
        return traces



class ReasoningExtractor:
    """
    Robust state-machine for extracting and filtering internal reasoning/planning tags from LLM streams.
    Handles split tags across network chunks and supports multiple tag pairs.
    """

    def __init__(self, tags_to_strip: Optional[List[str]] = None):
        self.tags_to_strip = tags_to_strip or ["thought", "think", "ice_plan"]
        self.buffer_state = 0  # 0 = Forwarding, 1 = Buffering
        self.current_tag = ""
        self.current_buffer = ""
        self.tag_detect_buffer = ""
        self.lookahead_size = 20  # Increased for longer tags like </ice_plan>

    def process_chunk(self, delta: str) -> tuple[str, list[tuple[str, str]]]:
        """
        Processes a chunk and returns (clean_text, list_of_extracted_tag_content).
        The list of extracted content is a list of (tag_name, content).
        """
        self.tag_detect_buffer += delta
        output_text = ""
        extracted = []

        while len(self.tag_detect_buffer) > 0:
            if self.buffer_state == 0:  # Forwarding
                # If buffer is too small to definitively say 'no tag', and we don't see one yet, wait
                if len(self.tag_detect_buffer) < self.lookahead_size:
                    could_be_tag = False
                    for tag in self.tags_to_strip:
                         if f"<{tag}>".startswith(self.tag_detect_buffer) or f"[{tag}]".startswith(self.tag_detect_buffer):
                             could_be_tag = True
                             break
                    if could_be_tag:
                         break # Need more data to be sure

                found_tag = None
                prefix_len = 0
                for tag in self.tags_to_strip:
                    if self.tag_detect_buffer.startswith(f"<{tag}>"):
                        found_tag = tag
                        prefix_len = len(tag) + 2
                    elif self.tag_detect_buffer.startswith(f"[{tag}]"):
                        found_tag = tag
                        prefix_len = len(tag) + 2
                    
                    if found_tag: break
                
                if found_tag:
                    self.buffer_state = 1
                    self.current_tag = found_tag
                    self.current_buffer = ""
                    self.tag_detect_buffer = self.tag_detect_buffer[prefix_len:]
                else:
                    output_text += self.tag_detect_buffer[0]
                    self.tag_detect_buffer = self.tag_detect_buffer[1:]
            else:  # Buffering
                actual_suffix = ""
                if self.tag_detect_buffer.startswith(f"</{self.current_tag}>"):
                    actual_suffix = f"</{self.current_tag}>"
                elif self.tag_detect_buffer.startswith(f"[/{self.current_tag}]"):
                    actual_suffix = f"[/{self.current_tag}]"
                
                if actual_suffix:
                    self.buffer_state = 0
                    extracted.append((self.current_tag, self.current_buffer.strip()))
                    self.tag_detect_buffer = self.tag_detect_buffer[len(actual_suffix):]
                    self.current_tag = ""
                    self.current_buffer = ""
                else:
                    # If buffer is too small to see a potential closer, wait
                    if len(self.tag_detect_buffer) < (len(self.current_tag) + 4):
                         break
                    self.current_buffer += self.tag_detect_buffer[0]
                    self.tag_detect_buffer = self.tag_detect_buffer[1:]

        return output_text, extracted

    def flush(self) -> tuple[str, list[tuple[str, str]]]:
        """
        Flushes any remaining content in the buffers at the end of the stream.
        """
        remaining = self.tag_detect_buffer
        self.tag_detect_buffer = ""
        
        output_text = ""
        extracted = []
        
        if self.buffer_state == 0:
            output_text = remaining
        else:
            # If we were buffering a tag, we expose it as "unfinished"
            unfinished = self.current_buffer + remaining
            if unfinished.strip():
                extracted.append((self.current_tag, unfinished.strip()))
            self.current_buffer = ""
            self.buffer_state = 0
            
        return output_text, extracted

