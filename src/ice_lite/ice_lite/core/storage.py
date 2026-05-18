import os
import logging
from typing import List
from pathlib import Path

logger = logging.getLogger("ice_storage")

class StorageManager:
    """
    Storage Abstraction Layer for ICE-Lite.
    Simplified to support only local filesystem paths.
    """

    @staticmethod
    def is_remote_path(path: str) -> bool:
        return False

    @staticmethod
    def is_dir(path: str) -> bool:
        """Determines if a path is a directory."""
        return os.path.isdir(path)

    @staticmethod
    def exists(path: str) -> bool:
        """Checks if a file or directory exists."""
        return os.path.exists(path)

    @staticmethod
    def list_files(path: str) -> List[str]:
        """Recursively lists all files in a directory."""
        if not os.path.isdir(path):
            return []
            
        files = []
        for root, _, filenames in os.walk(path):
            for filename in filenames:
                files.append(os.path.join(root, filename))
        return files

    @staticmethod
    def fetch_to_local(path: str, temp_dir: str) -> str:
        """For ICE-Lite, this just verifies the local file exists."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")
        return path

