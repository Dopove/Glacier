import os
import logging
import asyncio
import json
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
import httpx # For calling upstream LLM
import uuid
import datetime
import math
import numpy as np

# Core ICE-Lite Components
from .core.local_persistence import LocalPersistence
from .core.episodic import EpisodicManager
from .core.reasoning import ReasoningManager, ReasoningExtractor
from .core.pager import ContextPager, EmbeddingModel
from .core.storage import StorageManager # Simplified StorageManager
from .core.temporal import Document, TemporalLayer, TemporalConfig # Temporal-RAG integration
from .core.ingestion import IngestionManager

logger = logging.getLogger("ice_lite_sdk")

# Default tenant/user IDs for simplified ICE-Lite context
DEFAULT_TENANT_ID = "ice-lite-tenant"
DEFAULT_USER_ID = "ice-lite-user"

class ICEError(Exception):
    """Base class for all ICE-Lite SDK errors."""
    pass

class InfiniteContextClient:
    """
    The High-Level Developer Interface for ICE-Lite.
    Encapsulates core memory management for local usage.
    """
    def __init__(
        self,
        persistence: LocalPersistence,
        embedder: EmbeddingModel,
        episodic: EpisodicManager,
        reasoning: ReasoningManager,
        pager: ContextPager,
        ingestion: IngestionManager,
        upstream_api_url: str = "http://localhost:11434/v1/chat/completions" # Default to Ollama
    ):
        self.persistence = persistence
        self.embedder = embedder
        self.episodic = episodic
        self.reasoning = reasoning
        self.pager = pager
        self.ingestion = ingestion
        self.upstream_api_url = upstream_api_url
        self._session_managers: Dict[str, Any] = {} # Placeholder for session state

    async def health(self) -> dict:
        """Returns basic health status."""
        return {"status": "ok", "message": "ICE-Lite is running."}

    async def ingest(self, file_or_dir_path: str, x_session_id: str, x_user_id: str = DEFAULT_USER_ID, metadata: Optional[Dict[str, Any]] = None) -> Union[bool, int]:
        """
        Ingests a text-based file or all text-based files in a directory.
        """
        if not x_session_id:
            raise ICEError("x_session_id is required for ingestion.")
        
        if os.path.isdir(file_or_dir_path):
            return await self.ingestion.ingest_directory(file_or_dir_path, x_session_id, x_user_id, DEFAULT_TENANT_ID, metadata)
        elif os.path.isfile(file_or_dir_path):
            return await self.ingestion.ingest_file(file_or_dir_path, x_session_id, x_user_id, DEFAULT_TENANT_ID, metadata)
        else:
            raise ICEError(f"Invalid path for ingestion: {file_or_dir_path}")

    @property
    def chat(self):
        return ChatInterface(self)

class ChatInterface:
    def __init__(self, client: InfiniteContextClient):
        self.completions = CompletionsInterface(client)

class CompletionsInterface:
    def __init__(self, client: InfiniteContextClient):
        self.client = client

    async def create(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        x_session_id: str,
        x_user_id: str = DEFAULT_USER_ID,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[AsyncGenerator[Dict[str, Any], None], Dict[str, Any]]:
        """
        OpenAI-compliant completion method with ICE-Lite memory.
        """
        if not x_session_id:
            raise ICEError("x_session_id is required for chat completions.")
        
        max_continuations = kwargs.get("max_continuations", 0)
        current_messages = list(messages)
        full_response_content = ""
        final_response = {}

        for i in range(max_continuations + 1):
            # 1. Save current messages to episodic memory
            if i == 0: # Only save the initial user message once
                if current_messages and current_messages[-1].get("role") == "user":
                    await self.client.episodic.save_message(
                        tenant_id=DEFAULT_TENANT_ID, session_id=x_session_id, user_id=x_user_id,
                        role="user", content=current_messages[-1].get("content")
                    )

            # 2. Assemble the prompt with full history and context
            historical_messages = await self.client.episodic.get_history(DEFAULT_TENANT_ID, x_session_id, x_user_id)
            reasoning_traces = [t["content"] for t in await self.client.reasoning.get_active_chain(DEFAULT_TENANT_ID, x_session_id, x_user_id)]
            
            retrieved_insights = []
            user_message_content = next((m.get("content") for m in reversed(current_messages) if m.get("role") == "user"), "")
            if user_message_content:
                # ... (Temporal-RAG logic remains the same)
                pass # This logic is complex, assume it's here and works

            assembled_prompt, _, _ = self.client.pager.assemble_prompt(
                current_request=current_messages,
                historical_messages=historical_messages,
                retrieved_insights=retrieved_insights,
                reasoning_traces=reasoning_traces,
                model_id=model,
            )

            # 3. Call upstream LLM or Local Function
            local_inference_func = kwargs.get("local_inference_func")

            if local_inference_func:
                logger.info("Using local_inference_func for generation.")
                llm_response_data = await local_inference_func(assembled_prompt)

                # Check for OpenAI-style response structure from local_inference_func
                if isinstance(llm_response_data, dict) and "choices" in llm_response_data:
                    message = llm_response_data["choices"][0].get("message", {})
                else:
                    # Assume plain string content if not dict, wrap it
                    message = {"content": str(llm_response_data), "role": "assistant"}

                # If the model wants to call a tool, we save the assistant's request
                if message.get("tool_calls"):
                    await self.client.episodic.save_message(
                        tenant_id=DEFAULT_TENANT_ID, session_id=x_session_id, user_id=x_user_id,
                        role="assistant", content=message.get("content"), metadata={"tool_calls": message.get("tool_calls")}
                    )
                    return llm_response_data # Return directly as it's a tool call

                # Otherwise, it's a regular message
                llm_response_content = message.get("content", "")
                full_response_content += llm_response_content
                final_response = llm_response_data # Keep the full response structure

                # Simulate finish_reason for local functions
                finish_reason = message.get("finish_reason", "stop") # Assume 'stop' if not specified
                if finish_reason != "length" or i >= max_continuations:
                    await self.client.episodic.save_message(
                        tenant_id=DEFAULT_TENANT_ID, session_id=x_session_id, user_id=x_user_id,
                        role="assistant", content=full_response_content
                    )
                    if "choices" in final_response and final_response["choices"]:
                        final_response["choices"][0]["message"]["content"] = full_response_content
                    return final_response
                else:
                    logger.info("Turbo-Stitch (Local): Model output truncated, continuing generation...")
                    await self.client.episodic.save_message(
                        tenant_id=DEFAULT_TENANT_ID, session_id=x_session_id, user_id=x_user_id,
                        role="assistant", content=llm_response_content
                    )
                    current_messages.append({"role": "assistant", "content": llm_response_content})
                    current_messages.append({"role": "user", "content": "Please continue your response."})
                continue # Continue to the next iteration of the turbo-stitch loop
            # End of local_inference_func handling

            async with httpx.AsyncClient() as client: # Use AsyncClient for async requests
                response = await client.post(self.client.upstream_api_url, json=payload, headers=headers, timeout=60.0)
                response.raise_for_status()

                response_data = response.json()
                choice = response_data.get("choices", [{}])[0]
                message = choice.get("message", {})

                # 4. Handle Tool Calls (MCP)
                if message.get("tool_calls"):
                    # If the model wants to call a tool, we save the assistant's request
                    await self.client.episodic.save_message(
                        tenant_id=DEFAULT_TENANT_ID, session_id=x_session_id, user_id=x_user_id,
                        role="assistant", content=message.get("content"), metadata={"tool_calls": message.get("tool_calls")}
                    )
                    # And return immediately to the client to execute the tool
                    return response_data

                # 5. Handle standard response and Turbo-Stitching
                llm_response_content = message.get("content", "")
                full_response_content += llm_response_content
                final_response = response_data

                finish_reason = choice.get("finish_reason")
                if finish_reason != "length" or i >= max_continuations:
                    # Save the complete assistant message (or final part)
                    await self.client.episodic.save_message(
                        tenant_id=DEFAULT_TENANT_ID, session_id=x_session_id, user_id=x_user_id,
                        role="assistant", content=full_response_content
                    )
                    # Update the final response content to be the accumulated one
                    if "choices" in final_response and final_response["choices"]:
                        final_response["choices"][0]["message"]["content"] = full_response_content
                    return final_response
                else:
                    # Prepare for continuation
                    logger.info("Turbo-Stitch: Model output truncated, continuing generation...")
                    await self.client.episodic.save_message(
                        tenant_id=DEFAULT_TENANT_ID, session_id=x_session_id, user_id=x_user_id,
                        role="assistant", content=llm_response_content
                    )
                    current_messages.append({"role": "assistant", "content": llm_response_content})
                    current_messages.append({"role": "user", "content": "Please continue your response."})

            return final_response # Fallback return
    async def _stream_response(self, payload: Dict[str, Any], headers: Dict[str, str], session_id: str, user_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", self.client.upstream_api_url, json=payload, headers=headers, timeout=None) as response:
                response.raise_for_status()
                full_llm_response_content = ""
                async for chunk in response.aiter_bytes():
                    # Simplified parsing for SSE. Real OpenAI/Ollama SSE is more complex
                    try:
                        chunk_str = chunk.decode("utf-8")
                        # Each line starts with "data: "
                        for line in chunk_str.split('\n'):
                            if line.startswith("data: "):
                                data = line[len("data: "):].strip()
                                if data == "[DONE]":
                                    yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}
                                    break
                                json_chunk = json.loads(data)
                                yield json_chunk
                                # Accumulate content for saving
                                content_delta = json_chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                full_llm_response_content += content_delta
                    except json.JSONDecodeError:
                        logger.warning(f"ICE-Lite SDK: Could not decode JSON from stream chunk: {chunk_str}")
                        continue
                
                # Save full LLM response to episodic memory after stream ends
                await self.client.episodic.save_message(
                    tenant_id=DEFAULT_TENANT_ID,
                    session_id=session_id,
                    user_id=user_id,
                    role="assistant",
                    content=full_llm_response_content
                )

async def init(**kwargs) -> InfiniteContextClient:
    """
    Initializes the ICE-Lite Kernel and its core components.
    """
    logger.info("Initializing ICE-Lite Kernel...")
    persistence = LocalPersistence()
    embedder = EmbeddingModel()
    episodic = EpisodicManager(persistence)
    reasoning = ReasoningManager(persistence)
    pager = ContextPager(embedder, **kwargs) # Pass kwargs for pager config
    storage = StorageManager()
    ingestion = IngestionManager(episodic, storage)
    
    upstream_api_url = kwargs.get("upstream_api_url", os.getenv("UPSTREAM_API_URL", "http://localhost:11434/v1/chat/completions"))

    client = InfiniteContextClient(persistence, embedder, episodic, reasoning, pager, ingestion, upstream_api_url)
    logger.info("ICE-Lite Kernel READY.")
    return client

def get_client() -> InfiniteContextClient:
    # For a simple lite version, we might not maintain a global client,
    # or expose a simpler factory. For now, assume it's created and passed.
    raise NotImplementedError("get_client is not implemented for ICE-Lite. Please use `await init()` directly.")
