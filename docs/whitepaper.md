# GLACIER: Why Mamba Needs an External Hippocampus 

*And How We Fixed RAG's Blindness to Time*

---

Transformers have infinite intelligence but no memory. Mamba is blazingly fast but amnesiac. Today, we are releasing **GLACIER**—an architecture that gives State Space Models (SSMs) an external hippocampus, equipped with a temporal understanding of truth.

At the core of GLACIER are two powerful concepts:
1. **ICE-Lite:** An open-source, pure-Python Virtual Memory Manager.
2. **Temporal-RAG:** A time-aware post-retrieval reranker.

## The State Space Bottleneck: Context Rot

Mamba (and SSMs in general) achieve their incredible $O(N)$ scaling by compressing the entire sequence history into a fixed-size hidden state vector $h$ (e.g., `d_state=256`).

While this eliminates the massive $O(N^2)$ VRAM overhead of the Transformer's KV-cache, it introduces a fatal flaw for long-running agents: **Context Rot**.

Because the hidden state is a lossy compression, old information is continuously overwritten by new tokens based on Mamba's dynamic dwell time ($\Delta$). Once a session ends, the state is gone forever. If you start a new conversation, Mamba has no memory of the previous one. It suffers from *Amnesiac Amnesia*.

## Enter ICE-Lite: The External Hippocampus

To solve Context Rot, we must decouple active cognition from long-term storage. 

**ICE-Lite** wraps Mamba from the *outside*. Instead of forcing the SSM to memorize everything, ICE-Lite acts as a persistent episodic ledger.

1. **Memory Write:** After every turn, ICE-Lite saves the user query and the model's response (along with semantic embeddings) to a local JSON ledger.
2. **Memory Read:** Before the next turn, ICE-Lite semantically searches this ledger to find the most relevant past memories (the "needles") and prepends them to the Mamba prompt.

Mamba's 256-dimensional hidden state is now always operating on a fresh, highly-relevant prompt rather than a stale, decayed history.

## Fixing RAG's Blindness to Time

However, simply retrieving old memories introduces a new problem: **Stale Knowledge**. Standard RAG systems retrieve documents based on cosine similarity, ignoring *when* the information was true. A 2-year-old API policy that is a 95% semantic match will defeat yesterday's update that is a 90% match.

We integrated **Temporal-RAG** directly into ICE-Lite's retrieval phase to solve this.

Our retrieval pipeline now classifies memories into three states:
*   **VALID:** Permanently or openly true.
*   **TEMPORAL:** True only within a bounded time window (e.g., an outage).
*   **EXPIRED:** Was true, is no longer.

When ICE-Lite retrieves context, it performs a hybrid reranking:
`Final Score = Semantic Similarity + (Temporal Decay × Weight)`

Documents marked as `EXPIRED` are hard-removed. Memories age gracefully based on their kind (STATIC facts decay slower than VERSIONED policies). 

## The GLACIER Benchmark

In our "Context Rot Benchmark," we injected a critical "needle" into a conversation, followed by 50 turns of completely irrelevant "haystack" noise. 

A vanilla Mamba model's hidden state completely overwrote the needle. 

**GLACIER (Mamba + ICE-Lite + Temporal-RAG)** flawlessly retrieved the needle from its episodic ledger, scored its temporal validity, and injected it back into Mamba's prompt just-in-time for perfect recall.

---

*GLACIER is fully open-source (MIT). We invite the community to build persistent, time-aware agents.*
