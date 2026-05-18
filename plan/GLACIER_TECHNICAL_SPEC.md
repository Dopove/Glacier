# GLACIER Technical Specification: Mamba2 + ICE-Lite

## 1. Executive Summary
GLACIER is the integration of the **Mamba2** State Space Model (SSM) with **ICE-Lite** (Infinite Context Engine - Lite). 

**The Narrative:** Transformers have infinite intelligence but no memory (and scale quadratically $O(N^2)$). Mamba is blazingly fast ($O(N)$ time, $O(1)$ space) but suffers from fundamental amnesia due to its fixed-size state vector. 
**GLACIER solves this:** Mamba provides the fast working memory; ICE-Lite acts as the external hippocampus (persistent long-term memory).

## 2. The Core Problem: SSM Context Rot
Mamba compresses all sequence history into a fixed-size hidden state $h$ (e.g., `d_state=256`). This means:
*   **Lossy Compression:** Old information is continuously overwritten by new tokens based on the dwell time parameter ($\Delta$).
*   **Session-Local:** The state vanishes when the session ends.
*   **Non-Retrievable:** Cannot explicitly query past states.

This leads to inevitable **Context Rot** over long conversations.

## 3. The GLACIER Solution Architecture

ICE-Lite wraps Mamba from the *outside*, managing the context window before Mamba even sees the tokens. This ensures Mamba's 256-dim hidden state is always operating on a fresh, highly-relevant prompt rather than a stale, decayed history.

### Touchpoint 1: Memory Read (Temporal Just-In-Time Retrieval)
Before feeding input to Mamba, ICE-Lite retrieves relevant memories from previous turns.
*   **Component:** `ICE-Lite ContextPager` & `EmbeddingModel`
*   **Mechanism:** Semantically embeds the new user query, searches the `LocalPersistence` ledger, and prepends high-signal fragments (the "needles") to the prompt.
*   **Temporal Scoring (The "Temporal-RAG" Layer):** ICE-Lite incorporates a post-retrieval temporal reranking layer to combat both context rot and stale knowledge. Documents are scored using a hybrid formula: `Final Score = Semantic Similarity + (Temporal Decay × Weight)`.
    *   **Validity:** Documents marked as `EXPIRED` are hard-removed from the context.
    *   **Decay:** Memories age gracefully based on their `kind`. A `STATIC` fact (like a mathematical definition) decays very slowly, maintaining a high base score, whereas a `VERSIONED` fact or conversation turn decays based on a defined half-life.
    *   This ensures that Mamba prioritizes not just what is semantically similar, but what is *currently true and fresh*.

### Touchpoint 2: Context Pinning
Critical facts that must survive context rot indefinitely are pinned.
*   **Component:** `ICE-Lite ContextPager` (System/Floor messages)
*   **Mechanism:** Facts flagged as critical (e.g., `role: "system"`) are injected at the absolute top of the prompt window every single turn, ensuring Mamba's $\Delta$ parameter always absorbs them freshly.

### Touchpoint 3: Memory Write
After Mamba generates a response, the turn is permanently archived.
*   **Component:** `ICE-Lite EpisodicManager`
*   **Mechanism:** Saves the `(user_query, mamba_response)` pair, along with their vector embeddings, into the local JSON ledger (`LocalPersistence`) for future retrieval across any session.

## 4. GLACIER vs. Transformer + RAG

| Feature | Transformer + RAG | GLACIER (Mamba2 + ICE-Lite) |
| :--- | :--- | :--- |
| **In-context memory** | Full KV cache — $O(N^2)$ cost, huge VRAM | Mamba hidden state — $O(1)$ cost, tiny VRAM |
| **Long-term memory** | External vector DB bolted on | ICE-Lite — native session architecture |
| **Speed** | Slows down exponentially at long contexts | Stays **constant speed** regardless of history length |
| **Context rot** | Fails abruptly when context window fills | ICE scores and pins — mathematically controlled decay |
| **Session persistence** | Stateless by default | ICE persists across sessions natively |

## 5. Implementation Stack
*   **Inference Engine:** `mamba_ssm` (Apache 2.0)
*   **Memory Manager:** `ice_lite` (MIT)
    *   `EpisodicManager` (Storage)
    *   `ContextPager` (Prompt Assembly & Pinning)
    *   `EmbeddingModel` (ONNX all-MiniLM-L6-v2)
*   **Storage Substrate:** Local JSON / File System (`LocalPersistence`)
