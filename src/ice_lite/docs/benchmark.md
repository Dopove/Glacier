# GLACIER Benchmarks & Performance Metrics

This document outlines the empirical benchmarks designed to validate the **GLACIER** architecture (Mamba2 + ICE-Lite + Temporal-RAG), developed by **Saran S** at **Dopove Private Limited**.

The core objective of these tests is to prove that GLACIER successfully overcomes the fundamental limitations of State Space Models (SSMs)—specifically, context rot and session amnesia—without sacrificing their $O(1)$ speed advantages.

To run these tests yourself, execute:
```bash
python -m ice_lite.test_glacier
```

---

## 1. Context Rot Benchmark (The "Needle in a Haystack")

**The Problem:** Mamba's hidden state continuously overwrites itself. In a long conversation, early facts are "forgotten" as new tokens are processed.

**The Test:**
1.  **Establish the Needle:** We start a session and declare a critical fact (e.g., "The secret project codename is 'Project Glacier'.").
2.  **The Haystack (Context Drift):** We simulate a long passage of time and context drift by sending 50 turns of completely irrelevant "filler" conversation (e.g., facts about geography, math, random stories).
3.  **The Recall:** We ask the model: "What was the critical project codename I mentioned earlier?"

**The Results:**
*   **Vanilla Mamba:** Fails. By turn 50, the hidden state has entirely decayed the initial fact. The model hallucinates or admits it does not know.
*   **GLACIER:** **Passes.** ICE-Lite intercepts the query, semantically searches the episodic ledger, retrieves the "needle" from Turn 1 (applying temporal scoring to ensure validity), and injects it into Mamba's prompt window just-in-time. Mamba correctly answers "Project Glacier."

---

## 2. Token Efficiency Benchmark

**The Problem:** Standard Transformers prevent context rot by keeping the entire conversation history in their KV-cache. For a 50-turn conversation (averaging 50 tokens per turn), this requires processing and storing 2,500 tokens. This scales quadratically ($O(N^2)$), eventually causing Out-Of-Memory (OOM) errors and massive latency.

**The Test:** Calculate the token overhead required to achieve perfect recall at Turn 50.

**The Results:**
*   **Transformer (Full KV-Cache):** 2,500 tokens in the active context window.
*   **GLACIER:** ~250 tokens in the active context window.
*   **Verdict:** ICE-Lite only retrieves and injects the top-K relevant memories (e.g., the 5 most relevant previous turns). GLACIER achieves the same perfect recall as a Transformer while being **10x more token-efficient** at turn 50. This efficiency grows linearly as the conversation gets longer.

---

## 3. Cross-Session Persistence

**The Problem:** Mamba (like all raw models) is stateless. If the inference script stops, the session's hidden state is destroyed.

**The Test:**
1.  **Session A:** Initialize `ICE-Lite` and tell the model a fact (e.g., "The launch code is 8841-A-Delta.").
2.  **Destruction:** Completely destroy the `ICE-Lite` client and the Mamba model instance, simulating a server restart or the user closing the application.
3.  **Session B:** Initialize a brand new `ICE-Lite` client and Mamba model. Ask for the launch code, providing the original `session_id`.

**The Results:**
*   **Vanilla Mamba:** Fails. Amnesia is absolute upon restart.
*   **GLACIER:** **Passes.** The new ICE-Lite client connects to the `LocalPersistence` ledger, retrieves the fact saved during Session A, and successfully answers the query.
