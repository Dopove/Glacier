import asyncio
import os
import uuid
import json
from pathlib import Path
from typing import List, Dict, Any
import pytest
from unittest.mock import AsyncMock, patch

# Import the specific ICE-Lite components to be tested
from ice_lite.sdk import init as init_ice_lite, DEFAULT_TENANT_ID

# --- Test Configuration ---
TEST_SESSION_ID = "agentic-test-session"
TEST_USER_ID = "agentic-test-user"
TEMP_DIR = Path("./temp_test_files")

@pytest.fixture(scope="module", autouse=True)
def setup_teardown():
    TEMP_DIR.mkdir(exist_ok=True)
    yield
    import shutil
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    cache_dir = Path.home() / ".cache" / "ice_lite_data"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

@pytest.mark.asyncio
async def test_ingestion_and_recall():
    """Tests that the ingest() method correctly saves file content and that the chat() method can recall it."""
    ice_client = await init_ice_lite()
    file_content = "The critical configuration parameter is 'ENABLE_ASYNC_DISPATCH'."
    test_file = TEMP_DIR / "test_config.txt"
    with open(test_file, "w") as f:
        f.write(file_content)

    await ice_client.ingest(str(test_file), x_session_id=TEST_SESSION_ID, x_user_id=TEST_USER_ID)

    async def mock_generate(messages: List[Dict[str, Any]]) -> dict:
        prompt = json.dumps(messages) # Search entire context
        if "ENABLE_ASYNC_DISPATCH" in prompt:
            return {"choices": [{"message": {"content": "The parameter is 'ENABLE_ASYNC_DISPATCH'."}, "finish_reason": "stop"}]}
        return {"choices": [{"message": {"content": "I don't have that information."}, "finish_reason": "stop"}]}

    response = await ice_client.chat.completions.create(
        model="mock-model",
        messages=[{"role": "user", "content": "What is the critical configuration parameter?"}],
        x_session_id=TEST_SESSION_ID,
        x_user_id=TEST_USER_ID,
        local_inference_func=mock_generate
    )
    
    assert "ENABLE_ASYNC_DISPATCH" in response['choices'][0]['message']['content']

@pytest.mark.asyncio
async def test_turbo_stitching():
    """Tests that max_continuations correctly stitches a long response."""
    ice_client = await init_ice_lite()
    session_id = "turbo-stitch-test"

    call_count = 0
    async def mock_generate_truncated(messages: List[Dict[str, Any]]) -> dict:
        nonlocal call_count
        call_count += 1
        content = f"Part {call_count}."
        finish_reason = "length" if call_count < 3 else "stop"
        return {"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]}

    response = await ice_client.chat.completions.create(
        model="mock-stitch-model",
        messages=[{"role": "user", "content": "Tell a story."}],
        x_session_id=session_id,
        max_continuations=2,
        local_inference_func=mock_generate_truncated
    )

    final_content = response['choices'][0]['message']['content']
    assert "Part 1" in final_content
    assert "Part 2" in final_content
    assert "Part 3" in final_content
    assert call_count == 3

@pytest.mark.asyncio
async def test_tool_calling_memory():
    """Tests that tool call requests and tool results are correctly saved to memory."""
    ice_client = await init_ice_lite()
    session_id = "tool-call-test"
    
    # 1. Turn 1: Mock model requesting a tool call
    async def mock_request_tool(messages: List[Dict[str, Any]]) -> dict:
        return {
            "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "get_user", "arguments": "{}"}}]}}]
        }

    tool_call_response = await ice_client.chat.completions.create(
        model="mock-tool-model",
        messages=[{"role": "user", "content": "Get user"}],
        x_session_id=session_id,
        local_inference_func=mock_request_tool
    )
    
    tool_calls = tool_call_response['choices'][0]['message']['tool_calls']
    
    # 2. Turn 2: Providing tool result
    async def mock_final_answer(messages: List[Dict[str, Any]]) -> dict:
        return {"choices": [{"message": {"content": "User is Alice."}}]}

    await ice_client.chat.completions.create(
        model="mock-tool-model",
        messages=[
            {"role": "assistant", "tool_calls": tool_calls},
            {"role": "tool", "tool_call_id": "call_123", "name": "get_user", "content": "Alice"}
        ],
        x_session_id=session_id,
        local_inference_func=mock_final_answer
    )

    # 3. Verify memory
    history = await ice_client.episodic.get_history(DEFAULT_TENANT_ID, session_id, "ice-lite-user")
    assert any(msg['role'] == 'assistant' and 'tool_calls' in msg.get('metadata', {}) for msg in history)
    assert any(msg['role'] == 'tool' and 'Alice' in str(msg['content']) for msg in history)
