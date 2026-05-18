import os
import json
from typing import List, Dict, Any, Union
from pathlib import Path
import logging
import uuid
import datetime

logger = logging.getLogger("ice_local_persistence")

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, uuid.UUID):
            return str(obj)
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)

class LocalPersistence:
    """
    Manages local file-based storage for ICE-Lite.
    Stores episodic messages and reasoning traces in separate JSON files per session.
    """
    def __init__(self, base_dir: Union[str, Path] = "~/.cache/ice_lite_data"):
        self.base_dir = Path(base_dir).expanduser()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"LocalPersistence initialized with base directory: {self.base_dir}")

    def _get_session_dir(self, tenant_id: str, user_id: str, session_id: str) -> Path:
        """Generates a path for session-specific data."""
        # For ICE-Lite, tenant_id and user_id can be simplified to folders
        # For true "lite" behavior, we might even combine them or use only session_id
        # but for now, let's keep the structure for potential future expansion / clarity.
        session_path = self.base_dir / tenant_id / user_id / session_id
        session_path.mkdir(parents=True, exist_ok=True)
        return session_path

    def _get_file_path(self, session_dir: Path, category: str) -> Path:
        """Gets the file path for a specific category within a session directory."""
        return session_dir / f"{category}.json"

    def _load_data(self, file_path: Path) -> List[Dict[str, Any]]:
        """Loads data from a JSON file."""
        if not file_path.exists():
            return []
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                # Ensure UUIDs are re-hydrated if needed, or just keep as string
                return data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from {file_path}: {e}")
            return []

    def _save_data(self, file_path: Path, data: List[Dict[str, Any]]):
        """Saves data to a JSON file."""
        try:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2, cls=CustomJSONEncoder)
        except Exception as e:
            logger.error(f"Failed to save data to {file_path}: {e}")

    def save_messages(self, tenant_id: str, user_id: str, session_id: str, messages: List[Dict[str, Any]]):
        """Saves episodic messages for a session."""
        session_dir = self._get_session_dir(tenant_id, user_id, session_id)
        file_path = self._get_file_path(session_dir, "messages")
        self._save_data(file_path, messages)
        logger.debug(f"Saved {len(messages)} messages for session {session_id}")

    def load_messages(self, tenant_id: str, user_id: str, session_id: str) -> List[Dict[str, Any]]:
        """Loads episodic messages for a session."""
        session_dir = self._get_session_dir(tenant_id, user_id, session_id)
        file_path = self._get_file_path(session_dir, "messages")
        messages = self._load_data(file_path)
        logger.debug(f"Loaded {len(messages)} messages for session {session_id}")
        return messages

    def save_reasoning_traces(self, tenant_id: str, user_id: str, session_id: str, traces: List[Dict[str, Any]]):
        """Saves reasoning traces for a session."""
        session_dir = self._get_session_dir(tenant_id, user_id, session_id)
        file_path = self._get_file_path(session_dir, "reasoning_traces")
        self._save_data(file_path, traces)
        logger.debug(f"Saved {len(traces)} reasoning traces for session {session_id}")

    def load_reasoning_traces(self, tenant_id: str, user_id: str, session_id: str) -> List[Dict[str, Any]]:
        """Loads reasoning traces for a session."""
        session_dir = self._get_session_dir(tenant_id, user_id, session_id)
        file_path = self._get_file_path(session_dir, "reasoning_traces")
        traces = self._load_data(file_path)
        logger.debug(f"Loaded {len(traces)} reasoning traces for session {session_id}")
        return traces
