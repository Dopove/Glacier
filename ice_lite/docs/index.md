# GLACIER Documentation

Welcome to the documentation for **GLACIER** (Mamba2 + ICE-Lite + Temporal-RAG).

GLACIER is an open-source project by **Dopove Private Limited**, founded by **Saran S** ([@Saran-386](https://github.com/Saran-386)). It demonstrates how to give State Space Models (like Mamba) an external, persistent hippocampus, solving their inherent context rot problem.

## Table of Contents

1. [**Architecture**](architecture.md): Understand the theoretical foundation of GLACIER. Learn why Mamba suffers from context rot, how ICE-Lite acts as an external memory manager, and how Temporal-RAG fixes knowledge freshness.
2. [**Benchmark**](benchmark.md): Review the empirical results of our Context Rot, Token Efficiency, and Cross-Session Persistence tests.
3. [**API Reference**](api.md): Detailed documentation for the `ice_lite.sdk` and `PersistentMamba` classes.

## Getting Help
If you encounter any issues or have questions, please open an issue on the GitHub repository.

## Upgrading to Enterprise
GLACIER and ICE-Lite are designed for local prototypes and single-agent workflows. For multi-tenant isolation (RLS), Redis hot-caching, and hardware-bound security, consider upgrading to **ICE Enterprise**. Read the [Upgrade Guide](../../docs/ICE_vs_ICE-LITE.md) for more details.
