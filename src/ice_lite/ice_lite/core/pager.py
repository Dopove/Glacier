import os
import onnxruntime as ort
from tokenizers import Tokenizer
import numpy as np
import logging
import asyncio
from typing import Optional
from collections import deque
from pathlib import Path

logger = logging.getLogger("ice_pager")


class EmbeddingModel:
    """
    Local ONNX embedding model to guarantee sub-50ms latency.
    Uses 'all-MiniLM-L6-v2' exported to ONNX format.
    Models are expected to be present in ~/.cache/ice/models.
    """

    def __init__(self, model_dir: Optional[str] = None):
        # Default to global user cache if no directory provided
        if model_dir is None:
            model_dir = str(Path.home() / ".cache" / "ice" / "models")
        
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "model.onnx")
        self.tokenizer_path = os.path.join(model_dir, "tokenizer.json")

        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        try:
            # Explicitly check for model files
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"ONNX model not found at {self.model_path}. Please place 'model.onnx' there.")
            if not os.path.exists(self.tokenizer_path):
                raise FileNotFoundError(f"Tokenizer not found at {self.tokenizer_path}. Please place 'tokenizer.json' there.")

            # We load the tokenizer
            self.tokenizer = Tokenizer.from_file(self.tokenizer_path)
            # Create an ONNX inference session
            # Execution providers: adaptively select available providers
            available = ort.get_available_providers()
            providers = []
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")

            self.session = ort.InferenceSession(
                self.model_path,
                providers=providers,
            )
            logger.info(f"Loaded ONNX model from {self.model_path} using {self.session.get_providers()[0]}")
        except Exception as e:
            logger.warning(
                f"Failed to initialize embedding model: {e}"
            )
            raise # Re-raise to indicate a critical failure

    def embed(self, text: str) -> list[float]:
        """
        Embeds a single string and returns a vector (list of floats).
        """
        if not hasattr(self, "session"):
            # Return dummy if not loaded for testing
            return [0.0] * 384

        # [PHASE 29]: Handle multimodal/list content
        if isinstance(text, list):
            text = " ".join([str(item.get("text", "")) for item in text if isinstance(item, dict) and item.get("type") == "text"])
        
        if not text:
            return [0.0] * 384

    def embed(self, text: str) -> list[float]:
        """
        Embeds a single string and returns a vector (list of floats).
        """
        if not hasattr(self, "session"):
            # Return dummy if not loaded for testing
            return [0.0] * 384

        # [PHASE 29]: Handle multimodal/list content
        if isinstance(text, list):
            text = " ".join([str(item.get("text", "")) for item in text if isinstance(item, dict) and item.get("type") == "text"])

        if not text:
            return [0.0] * 384

        # Tokenize
        encoded = self.tokenizer.encode(text)

        # Prepare inputs according to all-MiniLM-L6-v2 ONNX schema usually expects:
        # input_ids, attention_mask, token_type_ids
        # ONNX models expect batch dimension: [batch_size, sequence_length]

        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

        # If model expects token_type_ids:
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        # We check expected inputs from the session
        expected_inputs = [i.name for i in self.session.get_inputs()]
        if "token_type_ids" in expected_inputs:
            inputs["token_type_ids"] = np.array([encoded.type_ids], dtype=np.int64)

        # Run inference
        outputs = self.session.run(None, inputs)

        # The output of MiniLM is generally the last hidden state [batch, seq_len, hidden_size]
        # We need to apply mean pooling
        token_embeddings = outputs[0]
        # Expand attention mask to match token embeddings shape
        input_mask_expanded = np.expand_dims(attention_mask, -1).astype(float)

        # Mean Pooling calculation
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)

        sentence_embeddings = sum_embeddings / sum_mask

        # L2 Normalize
        normalized = sentence_embeddings / np.linalg.norm(
            sentence_embeddings, axis=1, keepdims=True
        )

        return normalized[0].tolist()

    async def embed_async(self, text: str) -> list[float]:
        """
        Asynchronous wrapper for the ONNX embedding generation.
        Throws the heavy Numpy/ONNX operations onto a separate OS thread to avoid locking FastAPI's event loop.
        """
        return await asyncio.to_thread(self.embed, text)


class ContextPager:
    def __init__(self, embedder: EmbeddingModel, max_tokens: Optional[int] = None, max_input_tokens: Optional[int] = None, enable_smart_sizing: Optional[bool] = None, default_output_tokens: Optional[int] = None, **kwargs):
        self.embedder = embedder
        # Input/Output Tuning (Enterprise Controls)
        self.max_tokens = max_tokens or int(os.getenv("ICE_MODEL_CONTEXT_WINDOW", "8192"))
        self.max_input_tokens = max_input_tokens if max_input_tokens is not None else int(os.getenv("ICE_POST_COMPRESSION_LIMIT", str(self.max_tokens)))
        self.default_output_tokens = default_output_tokens if default_output_tokens is not None else int(os.getenv("ICE_DEFAULT_OUTPUT_TOKENS", "1024"))
        
        if enable_smart_sizing is not None:
            self.enable_smart_sizing = enable_smart_sizing
        else:
            self.enable_smart_sizing = os.getenv("ICE_DYNAMIC_OUTPUT_BUDGET", "true").lower() == "true"
            
        self.stitch_tail = kwargs.get("stitch_tail", int(os.getenv("ICE_STITCH_TAIL_CHARS", "2000")))

    def estimate_tokens(self, content) -> int:
        """Simple approximation using generators to avoid O(n) looping costs."""
        if isinstance(content, str):
            return len(content) // 4 + 1
        
        # Handle multimodal content
        if isinstance(content, list):
            total = 0
            for item in content:
                if not isinstance(item, dict): continue
                m_type = item.get("type")
                if m_type == "text":
                    total += len(item.get("text", "")) // 4 + 1
                elif m_type == "image_url":
                    total += 255 # Standard high-res image cost
                elif m_type == "input_audio":
                    # Audio usually consumes tokens based on duration (~20-50 tokens/sec)
                    # We use a conservative enterprise fallback for the budget
                    total += 512 
                elif m_type in ["video_url", "video"]:
                    # Video is typically sampled as multiple frames (e.g. 10-50 frames)
                    # Each frame acts as an image
                    total += 255 * 10 # Baseline for 10 sampled frames
            return total
            
        return 0

    def get_dynamic_budget(self, model_id: str) -> int:
        """
        [PHASE 22]: Dynamic Turn-Budgeting.
        Determines the sliding window size (context limit) for a single inference turn.
        """
        # If the engineer set a hard MAX_INPUT_TOKENS, we respect it regardless of model
        if os.getenv("ICE_POST_COMPRESSION_LIMIT"):
            return self.max_input_tokens

        model_lower = model_id.lower()
        
        # 1. Cloud Models (High Capacity)
        cloud_keywords = ["gpt-4", "gpt-3.5", "claude", "gemini", "sonar"]
        if any(k in model_lower for k in cloud_keywords):
            return 128000 # High context horizon for cloud
            
        # 2. Local Large Models (Mid Capacity)
        if any(k in model_lower for k in ["7b", "8b", "13b", "14b", "32b", "qwen3"]):
            return 32768
            
        # 3. Local Small Models (VRAM Friendly)
        if any(k in model_lower for k in ["3b", "1.5b", "0.5b", "ministral", "phi"]):
            return 8192
            
        # Fallback to base configuration
        return self.max_tokens

    def get_smart_output_budget(self, messages: list[dict]) -> int:
        """
        [PHASE 22]: Smart Output Sizing.
        Prioritizes the most recent intent (newest to oldest).
        """
        # Feature Toggle: If disabled, we return the hardcoded enterprise default
        if not self.enable_smart_sizing:
            return self.default_output_tokens

        detailed_keywords = ["detailed", "architectural", "complete", "comprehensive", "full", "explain", "analyze"]
        brief_keywords = ["brief", "summary", "tl;dr", "concise", "short"]

        # Iterate reverse to find newest intent first
        for msg in reversed(messages):
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join([str(item.get("text", "")) for item in content if isinstance(item, dict) and item.get("type") == "text"])
            
            content = str(content).lower()
            # PRIORITY: Detailed/Full overrides Brief/Concise in the same block
            if any(w in content for w in detailed_keywords):
                logger.info("SmartSizer: Detailed intent DETECTED (4096)")
                return 4096
            if any(w in content for w in brief_keywords):
                logger.debug("SmartSizer: Brief intent DETECTED (512)")
                return 512

        # Fallback for standard interaction
        return self.default_output_tokens

    def assemble_prompt(
        self,
        current_request: list[dict],
        historical_messages: list[dict],
        retrieved_insights: list[str],
        reasoning_traces: Optional[list[str]] = None,
        model_id: str = "ministral-3:3b",
        max_input_tokens: Optional[int] = None, # Allow dynamic override
    ) -> list[dict]:
        """
        Takes the pinned memory, retrieved insights, reasoning traces, and historical messages,
        and strict-truncates them to fit the `max_tokens` limit.
        """
        # 0. Apply Smart Token Sizing & Reservation
        total_budget = max_input_tokens or self.get_dynamic_budget(model_id)
        
        # [PHASE 23]: Calculate output reservation BEFORE filling prompt
        # Use full history + current request for intent detection
        output_budget = self.get_smart_output_budget(historical_messages + current_request)
        
        # The prompt itself must leave room for the output budget
        effective_limit = max(min(512, total_budget), total_budget - output_budget)
        
        final_prompt = []
        token_count = 0

        # 1. Identify System Prompt (Floor)
        # We look for the first system message in history or current_request
        system_msg = next((m for m in historical_messages + current_request if m.get("role") == "system"), None)
        
        if system_msg:
            # Augment existing system message
            content = system_msg["content"]
            if "ICE-backed" not in content:
                content = f"{content}\n\n[ENGINE]: High-context mode ACTIVE (ICE/15M). Maintain consistency."
            pinned = {"role": "system", "content": content}
        else:
            pinned = {
                "role": "system",
                "content": "You are ICE-backed Agent. Maintain deep reasoning and long-term consistency.",
            }
            
        pinned_tokens = self.estimate_tokens(pinned["content"])
        final_prompt.append(pinned)
        token_count += pinned_tokens

        # 2. Add current request (always prioritized)
        current_request_filtered = [m for m in current_request if m.get("role") != "system"]
        current_request_tokens = sum(
            self.estimate_tokens(m["content"]) for m in current_request_filtered
        )
        token_count += current_request_tokens


        # 3. Inject Reasoning Traces (High Priority for continuity)
        if reasoning_traces:
            reasoning_text = "\n".join(reasoning_traces)
            reasoning_msg = {
                "role": "system",
                "content": f"Active Reasoning Trace (Multi-step context):\n{reasoning_text}",
            }
            reasoning_tokens = self.estimate_tokens(reasoning_msg["content"])
            final_prompt.append(reasoning_msg)
            token_count += reasoning_tokens

        # 4. Inject Retrieved Insights (Deep Memory Context)
        if retrieved_insights:
            insight_text = "\n".join(retrieved_insights)
            insight_msg = {
                "role": "system",
                "content": (
                    "### DEEP MEMORY RETRIEVAL ###\n"
                    "The following fragments were retrieved from long-term memory based on the current context. "
                    "Use these for facts, code symbols, or logic mentioned in much older parts of the 15M context horizon:\n"
                    f"{insight_text}\n"
                    "### END DEEP MEMORY ###"
                ),
            }
            insight_tokens = self.estimate_tokens(insight_msg["content"])
            final_prompt.append(insight_msg)
            token_count += insight_tokens

        # 5. Inject Sliding Window (Chronological History)
        sliding_window: deque[dict] = deque()
        
        # Filter out system messages from history as they are handled in the floor
        history_filtered = [m for m in historical_messages if m.get("role") != "system"]

        for msg in reversed(history_filtered):
            msg_tokens = self.estimate_tokens(msg["content"])
            if token_count + msg_tokens > effective_limit:
                logger.info(f"Sliding window hitting limit ({effective_limit}), truncating older messages.")
                break

            sliding_window.appendleft(msg)
            token_count += msg_tokens

        final_prompt.extend(sliding_window)
        final_prompt.extend(current_request_filtered)

        logger.info(
            f"Assembled prompt with approx {token_count}/{total_budget} tokens. "
            f"Reserved {output_budget} for output breathing room ({model_id})."
        )
        return final_prompt, token_count, output_budget

