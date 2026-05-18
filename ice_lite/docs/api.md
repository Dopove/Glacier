# ICE-Lite API Reference

This document provides a detailed reference for the public-facing components of the **ICE-Lite** SDK, developed by **Dopove Private Limited**.

---

## 1. `ice_lite.sdk`

The main entry point for using ICE-Lite.

### `await init(**kwargs) -> InfiniteContextClient`
Initializes the ICE-Lite kernel and its core components (ContextPager, EpisodicManager, etc.) directly within your Python application process. This should be called *once* at application startup.

**Returns:** An initialized `InfiniteContextClient` instance.

---

## 2. `InfiniteContextClient`

The high-level developer interface for managing infinite memory.

### `chat.completions.create(...) -> Union[AsyncGenerator, Dict]`
An OpenAI-compliant completion method with integrated persistent memory.

**Parameters:**
*   `model` (str): The name of the model to use for inference.
*   `messages` (List[Dict]): A list of message objects, each containing a `role` and `content`.
*   `x_session_id` (str): **Mandatory.** A unique ID for the conversation thread.
*   `x_user_id` (str, optional): A unique ID for the user (for data isolation). Defaults to `ice-lite-user`.
*   `local_inference_func` (AsyncCallable, optional): If provided, ICE-Lite will call this local function with the assembled prompt instead of making an HTTP request.
*   `stream` (bool, optional): If true, returns an async generator for streaming the response.
*   `max_continuations` (int, optional): The maximum number of "Turbo-Stitch" recursive turns for long outputs. Defaults to 0.

### `await ingest(file_or_dir_path: str, x_session_id: str, x_user_id: str) -> Union[bool, int]`
Ingests a single text-based file or an entire directory into the session's episodic ledger.

**Returns:** 
*   If a file: `True` if successful, `False` otherwise.
*   If a directory: The total number of files successfully ingested.

---

## 3. `PersistentMamba` (Integration)

A wrapper class located in `ice_lite.integration` that simplifies the combination of Mamba2 and ICE-Lite.

### `__init__(model_name: str, ice_client: InfiniteContextClient)`
Initializes the class, loads the specified Mamba model from HuggingFace to the optimal device (CUDA or CPU), and binds it to the provided `ice_client`.

### `async chat(user_message: str, x_session_id: str, x_user_id: str)`
A high-level method to send a message to the persistent Mamba model. It automatically handles the memory search, temporal reranking, prompt assembly, and local model inference.
