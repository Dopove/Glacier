# GLACIER: Project Roadmap & Implementation Plan

This document outlines the phased engineering and product strategy for GLACIER, the integration of Mamba2, ICE-Lite, and Temporal-RAG. The goal is to establish Dopove as the leader in LLM memory architecture via open-source adoption, creating a massive top-of-funnel for ICE Enterprise.

---

## Phase 1: Foundation (The GLACIER Prototype)
*Status: Completed (Local Prototype)*

**Objective:** Prove the core physics of combining an $O(1)$ State Space Model with a persistent, time-aware virtual memory manager.

**Key Deliverables:**
1.  **ICE-Lite Core:** Stripped-down pure Python version of the Infinite Context Engine. Removed Postgres, Redis, and RLS dependencies. Implemented `LocalPersistence` (JSON-backed).
2.  **Mamba Integration:** Created `PersistentMamba` wrapper to intercept inference calls, enabling ICE-Lite to manage the prompt window *before* tokens hit the SSM state.
3.  **Temporal-RAG Layer:** Integrated time-aware scoring (decay, validity, kind classification) directly into the `ICE-Lite` retrieval phase to solve knowledge freshness.
4.  **Context Rot Benchmark:** Built `demo.py` to empirically prove GLACIER retains a "needle" across 50+ turns of "haystack" noise, whereas vanilla Mamba forgets.

---

## Phase 2: The Open Source Drop (Community Edition)
*Status: Next Up*

**Objective:** Package, polish, and publish the prototype to GitHub and PyPI to generate maximum developer mindshare and academic citations.

**Key Deliverables:**
1.  **Package Separation:** Cleanly separate `ice_lite` into its own distributable PyPI package (`pip install ice-lite`).
2.  **API Polish:** Ensure the `ice_lite.sdk` is a perfect 1:1 drop-in replacement for the OpenAI SDK, maximizing ease of adoption.
3.  **The "Context Rot" Paper/Blog:** Publish a highly technical blog post (or arxiv paper) detailing the architecture: "Why Mamba needs an External Hippocampus" and "Fixing RAG's Blindness to Time".
4.  **HuggingFace Spaces Demo:** Create a live Gradio/Streamlit demo showing the Context Rot Benchmark in real-time.

**Marketing Narrative:** *"Transformers have infinite intelligence but no memory. Mamba is blazingly fast but amnesiac. GLACIER gives them both a brain that understands time."*

---

## Phase 3: The Enterprise Bridge
*Status: Planning*

**Objective:** Monetize the open-source attention. Build the technical bridges that allow companies to seamlessly upgrade from the free local version to the paid, scalable enterprise version.

**Key Deliverables:**
1.  **Zero-Rewrite Migration:** Ensure that upgrading from `ice-lite` to the proprietary `ice-engine` wheel requires changing exactly one import line.
2.  **State Migration Tooling:** Provide a script (`ice migrate`) that converts local JSON histories (`LocalPersistence`) into the highly optimized PostgreSQL `episodic_ledger` format.
3.  **Enterprise Documentation:** Publish clear guides highlighting what ICE Enterprise provides over ICE-Lite (Row-Level Security, Redis Hot-Caching, Multi-Tenancy, Hardware Sandboxing, Sovereign Mode).

---

## Phase 4: Advanced Agentic Capabilities
*Status: Future Expansion*

**Objective:** Push the boundaries of what a persistent-memory agent can achieve autonomously.

**Key Deliverables:**
1.  **MCP (Model Context Protocol) Native Support:** Enhance `ICE-Lite` to natively parse, pin, and store MCP tool executions, allowing agents to interface with local filesystems and APIs without context overflow.
2.  **Multimodal Evolution:** Reintroduce lightweight, file-agnostic ingestion (images, PDFs) into `ICE-Lite` using open-source parsers, bridging the gap to ICE Enterprise's `RAGAnything` pipeline.
3.  **"Turbo-Stitching" for Mamba:** Experiment with using ICE to force Mamba models to generate extremely long outputs (10k+ words) by recursively saving and re-injecting the state.
4.  **Continuous Self-Improvement Loop:** Build scripts that automatically format `ICE-Lite` session JSONs into DPO/SFT datasets, allowing developers to fine-tune their Mamba models on their own agentic traces.
