import os
import logging
from typing import List, Optional
from .episodic import EpisodicManager
from .storage import StorageManager
from tqdm import tqdm

logger = logging.getLogger("ice_lite_ingestion")

class IngestionManager:
    """
    Manages lightweight ingestion of text-based files for ICE-Lite.
    """
    def __init__(self, episodic_manager: EpisodicManager, storage_manager: StorageManager):
        self.episodic = episodic_manager
        self.storage = storage_manager

    async def ingest_file(self, file_path: str, session_id: str, user_id: str, tenant_id: str):
        """
        Reads a text-based file and saves its content to episodic memory.
        """
        if not self.storage.exists(file_path):
            logger.error(f"File not found for ingestion: {file_path}")
            return False

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Save the entire file content as a single system message
            await self.episodic.save_message(
                tenant_id=tenant_id,
                session_id=session_id,
                user_id=user_id,
                role="system",
                content=content,
                metadata={"source_file": file_path}
            )
            logger.info(f"Successfully ingested {file_path} into session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to read or ingest file {file_path}: {e}")
            return False

    async def ingest_directory(self, dir_path: str, session_id: str, user_id: str, tenant_id: str):
        """
        Recursively ingests all text-based files from a directory.
        """
        if not self.storage.is_dir(dir_path):
            logger.error(f"Directory not found for ingestion: {dir_path}")
            return 0
        
        files_to_ingest = [
            f for f in self.storage.list_files(dir_path) 
            if f.endswith(('.txt', '.md', '.py', '.js', '.ts', '.html', '.css', '.json'))
        ]
        
        ingested_count = 0
        for file_path in tqdm(files_to_ingest, desc=f"Ingesting Directory '{dir_path}'"):
            if await self.ingest_file(file_path, session_id, user_id, tenant_id):
                ingested_count += 1
        
        logger.info(f"Ingested {ingested_count}/{len(files_to_ingest)} files from {dir_path}.")
        return ingested_count
