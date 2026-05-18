import logging
import uuid
import json
import os
from typing import List, Optional, Dict, Any, Union

from .local_persistence import LocalPersistence # Import LocalPersistence

logger = logging.getLogger("ice_episodic")

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
        # Serialize list content for storage
        db_content = content
        if isinstance(content, list):
            db_content = json.dumps(content)
        
        # Load existing messages
        messages = self.persistence_manager.load_messages(tenant_id, user_id, session_id)
        
        # Create new message entry
        new_msg = {
            "tenant_id": tenant_id,
            "session_id": session_id,
            "thread_id": thread_id,
            "role": role,
            "content": db_content,
            "embedding": embedding,
            "user_id": user_id,
            "metadata": metadata or {},
            "created_at": __import__('datetime').datetime.now().isoformat()
        }
        
        # Append and save
        messages.append(new_msg)
        self.persistence_manager.save_messages(tenant_id, user_id, session_id, messages)

    async def get_history(
        self, tenant_id: str, session_id: str, user_id: str = "default-user", limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Retrieves chronological history for a session from local storage.
        """
        raw_msgs = self.persistence_manager.load_messages(tenant_id, user_id, session_id)
        
        # In a real database we'd query with LIMIT, here we slice the list
        if limit and len(raw_msgs) > limit:
             raw_msgs = raw_msgs[-limit:]

        processed = []
        for m in raw_msgs:
            content = m["content"]
            if isinstance(content, str) and (content.startswith("[") or content.startswith("{")):
                try:
                    content = json.loads(content)
                except: pass
            processed.append({"role": m["role"], "content": content})
        return processed


