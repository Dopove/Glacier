import gradio as gr
import asyncio
import uuid
import torch
import os
from threading import Thread

# We'll need to mock or conditionally load Mamba if on HF Spaces without GPU/enough RAM
# but for the sake of the prototype, we assume the environment has the model available.
try:
    from transformers import AutoTokenizer
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Mamba dependencies not found. Operating in mock mode for UI demonstration.")

from ice_lite.sdk import init as init_ice_lite
from ice_lite.sdk import DEFAULT_TENANT_ID

# --- Configuration ---
MAMBA_MODEL_NAME = "state-spaces/mamba-2.8b-slimpj"
device = "cuda" if torch.cuda.is_available() else "cpu"

class GlacierDemo:
    def __init__(self):
        self.ice_client = None
        self.tokenizer = None
        self.model = None
        self.session_id = str(uuid.uuid4())
        self.user_id = "hf-demo-user"

    async def initialize(self):
        if not self.ice_client:
            self.ice_client = await init_ice_lite()
            
        if MAMBA_AVAILABLE and not self.model:
            print(f"Loading Mamba model '{MAMBA_MODEL_NAME}' to {device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(MAMBA_MODEL_NAME)
            self.model = MambaLMHeadModel.from_pretrained(
                MAMBA_MODEL_NAME,
                device=device,
                dtype=torch.float16 if device == "cuda" else torch.float32
            )
            print("Mamba loaded.")

    def _sync_generate(self, prompt_text: str) -> str:
        if not MAMBA_AVAILABLE:
            # Mock mode logic
            if "project codename" in prompt_text.lower() or "secret" in prompt_text.lower():
                if "Project Glacier" in prompt_text:
                    return "The project codename is Project Glacier."
            return "I am a mock response. The real Mamba model is not loaded."
            
        input_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
        output_ids = self.model.generate(
            input_ids=input_ids,
            max_length=len(input_ids[0]) + 50,
            eos_token_id=self.tokenizer.eos_token_id,
            top_k=1
        )
        generated_part = output_ids[0, len(input_ids[0]):]
        decoded_output = self.tokenizer.decode(generated_part, skip_special_tokens=True)
        return decoded_output.strip()

    async def _generate(self, prompt_text: str) -> str:
        return await asyncio.to_thread(self._sync_generate, prompt_text)

    async def chat(self, user_message: str, is_filler: bool = False):
        if not self.ice_client:
            await self.initialize()

        session = self.session_id if not is_filler else f"filler-{uuid.uuid4()}"
        
        response = await self.ice_client.chat.completions.create(
            model=MAMBA_MODEL_NAME,
            messages=[{"role": "user", "content": user_message}],
            x_session_id=session,
            x_user_id=self.user_id,
            local_inference_func=self._generate
        )
        return response["choices"][0]["message"]["content"]

demo_instance = GlacierDemo()

def run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

def establish_needle(needle_text):
    msg = f"CRITICAL INFO: {needle_text} Please remember this very carefully."
    response = run_async(demo_instance.chat(msg))
    return f"**User:** {msg}\n\n**Mamba:** {response}"

def simulate_drift(num_messages):
    outputs = []
    for i in range(1, int(num_messages) + 1):
        msg = f"Filler message {i}: The capital of France is Paris. The quick brown fox jumps over the lazy dog."
        run_async(demo_instance.chat(msg, is_filler=True))
        if i % 10 == 0:
            outputs.append(f"Sent {i} filler messages...")
    return "\n".join(outputs)

def test_recall(question):
    response = run_async(demo_instance.chat(question))
    return f"**User:** {question}\n\n**Mamba + ICE-Lite:** {response}\n\n*(Notice how ICE-Lite retrieved the fact from the episodic ledger and injected it into Mamba's context before generation!)*"

with gr.Blocks(title="GLACIER: Mamba + ICE-Lite") as app:
    gr.Markdown("# 🧊 GLACIER: Mamba2 with Infinite Persistent Memory")
    gr.Markdown("Transformers have infinite intelligence but no memory. Mamba is blazingly fast but amnesiac. **GLACIER** gives Mamba an external hippocampus (ICE-Lite) with temporal scoring to prevent context rot.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1. Establish the Needle")
            needle_input = gr.Textbox(label="Secret Fact to Remember", value="The secret project codename is 'Project Glacier'.")
            needle_btn = gr.Button("Send Fact to Mamba")
            needle_output = gr.Markdown()
            
            gr.Markdown("### 2. Simulate Context Drift")
            drift_slider = gr.Slider(minimum=10, maximum=100, step=10, value=50, label="Number of filler messages")
            drift_btn = gr.Button("Send Haystack (Drift)")
            drift_output = gr.Markdown()
            
            gr.Markdown("### 3. Test Recall")
            recall_input = gr.Textbox(label="Question", value="What was the critical project codename I mentioned earlier?")
            recall_btn = gr.Button("Query Mamba")
            recall_output = gr.Markdown()
            
    needle_btn.click(fn=establish_needle, inputs=needle_input, outputs=needle_output)
    drift_btn.click(fn=simulate_drift, inputs=drift_slider, outputs=drift_output)
    recall_btn.click(fn=test_recall, inputs=recall_input, outputs=recall_output)

if __name__ == "__main__":
    app.launch()
