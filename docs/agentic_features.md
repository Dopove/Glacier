# Advanced Agentic Features in GLACIER

GLACIER is more than just a memory layer; it's a substrate for building sophisticated autonomous agents. The `ICE-Lite` SDK includes powerful features designed to handle multi-step tool use, long-form generation, and knowledge ingestion.

This guide, authored by **Saran S** of **Dopove Private Limited**, explains how to leverage these advanced capabilities.

---

## 1. Tool Calling & MCP Support

GLACIER solves "Agentic Amnesia" by treating tool interactions as a core part of its memory. When your agent uses a tool, ICE-Lite automatically saves the `tool_calls` request and the corresponding `tool` result to the episodic ledger.

This ensures that on subsequent turns, the agent remembers the outcome of its previous actions.

**Example Workflow:**

```python
# 1. Define your tools (OpenAI format)
tools = [{"type": "function", "function": {"name": "get_weather", ...}}]

# 2. Agent decides to use a tool
response = await ice_client.chat.completions.create(
    model="mamba-2.8b",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
    x_session_id="agent-weather-task"
)
# -> ICE-Lite returns the tool_calls payload and saves the assistant's intent.

# 3. You execute the tool and return the result
weather_result = '{"temp": "22C"}'
response = await ice_client.chat.completions.create(
    model="mamba-2.8b",
    messages=[
        {"role": "assistant", "tool_calls": response.choices[0].message.tool_calls},
        {"role": "tool", "tool_call_id": ..., "name": "get_weather", "content": weather_result}
    ],
    x_session_id="agent-weather-task"
)
# -> ICE-Lite saves the tool result and the agent gives the final answer.
```

## 2. Turbo-Stitching for Infinite Output

SSMs and LLMs have a `max_tokens` limit on their output. **Turbo-Stitching** allows GLACIER to bypass this. If a model's generation is cut off, ICE-Lite can automatically re-prompt it to "continue," seamlessly stitching the outputs together.

To use it, pass the `max_continuations` parameter:

```python
response = await ice_client.chat.completions.create(
    model="mamba-2.8b",
    messages=[{"role": "user", "content": "Write a 10,000-word story."}],
    x_session_id="long-story-session",
    max_continuations=20 # Allow up to 20 recursive turns
)

# The final response['choices'][0]['message']['content'] will be the full, stitched story.
```

## 3. Lightweight Multimodal Ingestion

You can give your Mamba agent knowledge of external files or even entire codebases with a single command.

The `ingest()` method supports both individual files and directories.

**Example: Ingesting a single file**
```python
await ice_client.ingest(
    file_or_dir_path="./project/config.py",
    x_session_id="codebase-analysis-session"
)
```

**Example: Ingesting an entire directory**
```python
# ICE-Lite will recursively find and ingest all supported text files.
await ice_client.ingest(
    file_or_dir_path="./project/src/",
    x_session_id="codebase-analysis-session"
)
```

After ingestion, you can ask questions about the content, and ICE-Lite's Temporal-RAG layer will retrieve the relevant snippets for Mamba to use.
