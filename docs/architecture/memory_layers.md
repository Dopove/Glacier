# The Multi-Tiered Memory Horizon: GLACIER's Data Management

GLACIER's core strength lies in its ability to manage vast amounts of contextual data across multiple tiers, providing both speed for active cognition and persistence for long-term recall. This architecture allows Mamba to operate with a "100 Billion Token Horizon."

**[IMAGE PLACEHOLDER: GLACIER Multi-Tiered Memory Hierarchy]**

*   **Diagram Description:**
    *   **Title:** GLACIER's Multi-Tiered Memory Hierarchy
    *   **Layout:** A vertical stack, representing different tiers of memory.
    *   **Components (from top to bottom):**
        1.  **Mamba SSM (Active State):** At the very top, representing the fastest, smallest, and most ephemeral memory.
            *   Label: "Mamba's O(1) Internal State (Transient, fixed-size)"
            *   Size: "~256 Dimensions (d_state)"
        2.  **ICE-Lite (Context Window):** Immediately below Mamba, representing the dynamic, in-RAM context assembled for each inference turn.
            *   Label: "ICE-Lite Context Window (Dynamic, <= 512 tokens)"
            *   Size: "~512 Tokens (Active RAM for current turn)"
            *   Key action: "Precision Paging & Stitching"
        3.  **Hot Cache (Redis - Future Phase):** Below ICE-Lite. This tier is part of the enterprise version but concept can be introduced.
            *   Label: "Hot Cache (Redis - Future: Sub-10ms Recall)"
            *   Size: "~1M Tokens (Fastest-access recent memory)"
        4.  **Semantic Ledger (LocalPersistence):** The largest, most persistent tier, stored on disk.
            *   Label: "Semantic Ledger (LocalPersistence: 100 Billion Token Horizon)"
            *   Size: "~100 Billion Tokens (Disk-based long-term memory)"
            *   Key action: "Episodic Storage, Temporal Metadata"
    *   **Flow/Arrows:**
        *   Vertical arrows connecting tiers, showing data movement up (retrieval) and down (storage).
        *   Arrows from "Semantic Ledger" -> "ICE-Lite Context Window": "Relevant Context Paged In"
        *   Arrows from "ICE-Lite Context Window" -> "Semantic Ledger": "New Memory Stored"
    *   **Key Callouts:**
        *   Overall diagram: "Memory Horizon: O(1) to 100 Billion Tokens"
        *   Each tier: Speed (e.g., "Nanoseconds", "Milliseconds", "Seconds"), Capacity (e.g., "Tiny", "Small", "Large", "Vast").

## Memory Hierarchy Explained

GLACIER treats memory as a multi-tiered hierarchy, ensuring optimal performance for Mamba while providing infinite recall.

1.  **Mamba's Internal State:** This is the model's fixed-size working memory, optimized for ultra-fast, local computation. GLACIER respects this $O(1)$ internal state by feeding it highly-concentrated, relevant prompts.
2.  **ICE-Lite Context Window:** For each turn, ICE-Lite dynamically assembles a context-rich prompt in RAM. This window is kept small and optimized, acting as a highly efficient L1 cache for Mamba.
3.  **Semantic Ledger (LocalPersistence):** This is GLACIER's long-term memory. It's a disk-based store for all historical interactions and ingested documents, allowing for a virtually infinite memory horizon. Each piece of memory is stored with rich temporal metadata, enabling Temporal-RAG.

This tiered approach ensures that GLACIER agents have both lightning-fast "working memory" and a comprehensive, persistent "life history" at their disposal.
