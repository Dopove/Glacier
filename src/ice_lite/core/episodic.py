import logging
import uuid
import json
import os
from typing import List, Optional, Dict, Any, Union

from .local_persistence import LocalPersistence # Import LocalPersistence

logger = logging.getLogger("ice_episodic")
logger.setLevel(logging.DEBUG)

class EpisodicManager:
    """
    Manages the episodic ledger for ICE, storing and retrieving 
    full conversational history across sessions/threads.
    """

    def __init__(self, persistence_manager: LocalPersistence):
        self.persistence_manager = persistence_manager

    async def save_message(
        self,
        tenant_id: str,
        session_id: str,
        role: str,
        content: Union[str, List[Any]],
        thread_id: str = "default",
        user_id: str = "default-user",
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ):
        """
        Saves a single message to the episodic_ledger.
        """
        logger.debug(f"Episodic: Saving message for session={session_id}, role={role}, content_len={len(str(content))}")
        
        # Load existing messages
        messages = self.persistence_manager.load_messages(tenant_id, user_id, session_id)
        
        # Create new message entry
        new_msg = {
            "tenant_id": tenant_id,
            "session_id": session_id,
            "thread_id": thread_id,
            "role": role,
            "content": content,
            "embedding": embedding,
            "user_id": user_id,
            "metadata": metadata or {}, # Ensure metadata is stored
            "created_at": __import__('datetime').datetime.now().isoformat()
        }
        
        # Append and save
        messages.append(new_msg)
        self.persistence_manager.save_messages(tenant_id, user_id, session_id, messages)
        logger.debug(f"Episodic: Message saved. Total messages in session={len(messages)}")

    async def get_history(
        self, tenant_id: str, session_id: str, user_id: str = "default-user", limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Retrieves chronological history for a session from local storage.
        """
        logger.debug(f"Episodic: Retrieving history for session={session_id}, user={user_id}")
        raw_msgs = self.persistence_manager.load_messages(tenant_id, user_id, session_id)
        logger.debug(f"Episodic: Found {len(raw_msgs)} raw messages.")
        
        # In a real database we'd query with LIMIT, here we slice the list
        if limit and len(raw_msgs) > limit:
             raw_msgs = raw_msgs[-limit:]

        processed = []
        for m in raw_msgs:
            # We must return role, content, AND metadata for things like tool calls to work
            # Ensure 'metadata' key is always present, even if empty in stored message
            content_val = m.get("content")
            if isinstance(content_val, dict): # For cases where tool_calls might be directly in content
                content_val = json.dumps(content_val) # Convert dict content to string for consistency
            content_val = content_val if content_val is not None else "" # Ensure it's not None

            processed_message = {
                "role": m["role"], 
                "content": content_val,
                "metadata": m.get("metadata", {}) # Correctly retrieve metadata if present, else empty dict
            }
            # The 'tool_calls' might be stored under metadata.tool_calls or in content field directly
            # depending on how the upstream LLM returns it. Normalize it to metadata.tool_calls.
            if m.get("tool_calls"): # If directly in message object (older format)
                if "tool_calls" not in processed_message["metadata"]:
                    processed_message["metadata"]["tool_calls"] = m["tool_calls"]
            elif isinstance(m.get("content"), dict) and m["content"].get("tool_calls"): # If in content (OpenAI format)
                if "tool_calls" not in processed_message["metadata"]:
                    processed_message["metadata"]["tool_calls"] = m["content"]["tool_calls"]

            processed.append(processed_message)
        logger.debug(f"Episodic: Returning {len(processed)} processed messages.")
        return processed