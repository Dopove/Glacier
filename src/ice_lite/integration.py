# integration.py
import asyncio
import os
from typing import List, Dict, Any

# Ensure torch, transformers, and mamba_ssm are installed
# pip install torch transformers mamba_ssm causal_conv1d
try:
    import torch
    from transformers import AutoTokenizer
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
except ImportError as e:
    print(f"Missing dependencies: {e}. Please run 'pip install torch transformers mamba_ssm causal_conv1d'.")
    exit(1)

from .sdk import init as init_ice_lite
from .sdk import InfiniteContextClient, DEFAULT_TENANT_ID # Import DEFAULT_TENANT_ID

# --- Configuration ---
MAMBA_MODEL_NAME = "state-spaces/mamba-2.8b-slimpj" # Example model
SESSION_ID = "mamba-test-session"

class PersistentMamba:
    """
    A wrapper that combines a Mamba model with the ICE-Lite SDK
    to provide persistent, long-term memory.
    """
    def __init__(self, model_name: str, ice_client: InfiniteContextClient):
        self.model_name = model_name
        self.ice_client = ice_client
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading Mamba model '{model_name}' to {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = MambaLMHeadModel.from_pretrained(
            model_name,
            device=self.device,
            dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        print("Mamba model loaded successfully.")

    def _sync_generate(self, prompt_text: str) -> str:
        """
        Synchronous helper for running Mamba generation.
        """
        input_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(self.device)
        
        # Mamba's generate function
        output_ids = self.model.generate(
            input_ids=input_ids,
            max_length=len(input_ids[0]) + 50, # Generate up to 50 new tokens
            eos_token_id=self.tokenizer.eos_token_id,
            top_k=1 # Greedy decoding
        )
        
        # Decode only the generated part
        generated_part = output_ids[0, len(input_ids[0]):]
        decoded_output = self.tokenizer.decode(generated_part, skip_special_tokens=True)
        return decoded_output.strip()

    async def _generate(self, prompt_text: str) -> str:
        """
        Async wrapper for Mamba generation to be used by ICE-Lite.
        """
        return await asyncio.to_thread(self._sync_generate, prompt_text)

    async def chat(self, user_message: str, x_session_id: str = SESSION_ID, x_user_id: str = "default-user"):
        """
        Initiates a chat turn with the persistent Mamba model.
        """
        # Only print user message for primary session to avoid spamming for filler messages
        if x_session_id == SESSION_ID: 
            print(f"\nUSER: {user_message}")
        
        response = await self.ice_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": user_message}],
            x_session_id=x_session_id,
            x_user_id=x_user_id,
            local_inference_func=self._generate # Pass the local Mamba generator
        )
        
        assistant_response = response["choices"][0]["message"]["content"]
        if x_session_id == SESSION_ID:
            print(f"MAMBA: {assistant_response}")
        return assistant_response

async def main():
    print("--- Mamba + ICE-Lite Integration Demo ---")
    
    # 1. Initialize ICE-Lite
    ice_lite_client = await init_ice_lite()

    # 2. Initialize PersistentMamba
    try:
        persistent_mamba = PersistentMamba(MAMBA_MODEL_NAME, ice_lite_client)
    except Exception as e:
        print(f"Failed to initialize PersistentMamba: {e}")
        return

    # 3. Demonstrate a multi-turn conversation and Context Rot prevention
    user_id = "demo-user"

    print("\n--- PHASE 1: Establishing the Needle ---")
    needle = "The secret project codename is 'Project Glacier'."
    await persistent_mamba.chat(f"CRITICAL INFO: {needle} Please remember this very carefully.", x_user_id=user_id)
    
    print("\n--- PHASE 2: Simulating Context Drift (Haystack) ---")
    print("Sending 50 irrelevant messages to simulate context rot for a vanilla model...")
    for i in range(1, 51):
        # We save this as a separate session to not pollute the primary one but simulate time passing
        await persistent_mamba.chat(f"Filler message {i}: This is just some random text about the weather or current events that is completely unrelated to anything important. The capital of France is Paris. What is 2+2? The quick brown fox jumps over the lazy dog. A rolling stone gathers no moss.", x_user_id=user_id, x_session_id=f"filler-session-{i}")
        if i % 10 == 0:
            print(f"  ... {i} messages sent ...")

    print("\n--- PHASE 3: Testing Recall (Needle Retrieval) ---")
    recall_question = "What was the critical project codename I mentioned earlier?"
    print(f"\nUSER: {recall_question}")

    response_recall = await persistent_mamba.ice_client.chat.completions.create(
        model=persistent_mamba.model_name,
        messages=[{"role": "user", "content": recall_question}],
        x_session_id=SESSION_ID,
        x_user_id=user_id,
        local_inference_func=persistent_mamba._generate
    )
    
    mamba_recall_response = response_recall["choices"][0]["message"]["content"]
    print(f"MAMBA: {mamba_recall_response}")

    if needle in mamba_recall_response:
        print(f"\n✅ VERDICT: Mamba + ICE-Lite successfully recalled the needle: '{needle}'. Context rot prevented!")
    else:
        print(f"\n❌ VERDICT: Mamba + ICE-Lite FAILED to recall the needle. Context rot detected or integration issue.")

    print("\n--- Demo Complete ---")
    print(f"Conversation history is saved in ~/.cache/ice_lite_data/{DEFAULT_TENANT_ID}/{user_id}/{SESSION_ID}/messages.json")
    print(f"Filler messages saved in ~/.cache/ice_lite_data/{DEFAULT_TENANT_ID}/{user_id}/filler-session-X/messages.json")


if __name__ == "__main__":
    # To run this, you need to be in the parent directory of ICE/
    # and run as a module: python -m ICE.ice_lite.integration
    asyncio.run(main())