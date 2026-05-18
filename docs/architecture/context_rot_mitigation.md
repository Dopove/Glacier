# Defeating Context Rot: A Visual Comparison

"Context Rot" is the fundamental flaw of State Space Models (SSMs) like Mamba. Over a long conversation, the model's fixed-size hidden state gradually forgets earlier information as it processes new tokens. 

GLACIER's architecture is explicitly designed to solve this problem, ensuring perfectly stable recall regardless of conversation length.

**[IMAGE PLACEHOLDER: Context Rot vs. GLACIER Flow]**

*   **Diagram Description:**
    *   **Title:** Context Rot: Vanilla Mamba vs. GLACIER
    *   **Layout:** Two side-by-side vertical flowcharts or timelines.
    *   **Left Side (Vanilla Mamba - "The Problem"):**
        1.  **Time $T=0$:** A sharp, clear block representing the initial fact: "Fact A: My name is Saran."
        2.  **Time $T=1 \dots T=49$ (The Haystack):** A long, scrolling list of irrelevant filler text (e.g., "Filler data...", "Unrelated query...").
        3.  **The Hidden State (Visualized):** Alongside the timeline, show a block representing Mamba's hidden state. As time progresses from $T=1$ to $T=49$, visually show "Fact A" fading away, dissolving, or being overwritten by the "Filler data."
        4.  **Time $T=50$ (The Query):** "What is my name?"
        5.  **Result (Red Cross):** "FAILURE: The hidden state has decayed. Mamba forgets."
    *   **Right Side (GLACIER - "The Solution"):**
        1.  **Time $T=0$:** The initial fact is stated: "Fact A: My name is Saran." -> Arrow pointing to a solid vault/ledger icon labeled "Saved to Semantic Ledger."
        2.  **Time $T=1 \dots T=49$ (The Haystack):** The same long, scrolling list of filler text.
        3.  **The Semantic Ledger (Visualized):** Alongside the timeline, show the ledger. "Fact A" remains solid and perfectly intact inside the vault throughout the filler turns.
        4.  **Time $T=50$ (The Query):** "What is my name?" -> Arrow points to the ICE-Lite engine.
        5.  **ICE-Lite Action:** "ICE-Lite intercepts query." -> Arrow to Semantic Ledger -> "Retrieves Fact A."
        6.  **Prompt Assembly:** ICE-Lite dynamically builds a prompt containing both "Fact A" and the query.
        7.  **Result (Green Check):** "SUCCESS (100% Recall): Mamba infers from a freshly injected prompt."
    *   **Key Takeaway Banner:** "GLACIER decouples memory storage from active inference, eliminating the continuous decay of information."

## Why the Hidden State Fails (and Why ICE-Lite Succeeds)

Mamba achieves its blazing speed by compressing all history into a fixed vector (e.g., 256 dimensions). This is an inherently lossy compression. Older information is "pushed out" to make room for new tokens.

GLACIER recognizes that **long-term memory cannot be solved by active state alone.**

By intercepting the conversation and storing every turn in the **Semantic Ledger**, GLACIER preserves information losslessly. When a new query arrives, ICE-Lite performs a lightning-fast semantic search (enhanced by Temporal-RAG to ensure validity), retrieves the precise "needle" from the "haystack," and feeds it to Mamba as fresh context.

This ensures Mamba always operates with perfect, un-decayed memory, no matter how long the session runs.
