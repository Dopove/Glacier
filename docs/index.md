# GLACIER Documentation

Welcome to the documentation for **GLACIER** (Mamba2 + ICE-Lite + Temporal-RAG).

GLACIER is an open-source project by **Dopove Private Limited**, founded by **Saran S** ([@Saran-386](https://github.com/Saran-386)). It demonstrates how to give State Space Models (like Mamba) an external, persistent hippocampus, solving their inherent context rot problem while maintaining temporal awareness.

## Table of Contents

1.  [**Getting Started**](getting_started.md): Installation instructions and a guide to downloading required embedding models.
2.  [**Architecture**](architecture.md): Understand the theoretical foundation of GLACIER.
3.  [**Benchmark**](benchmark.md): Review the empirical results of our core memory benchmarks.
4.  [**Agentic Features**](agentic_features.md): Learn how to use Tool-Calling, Turbo-Stitching, and Ingestion.
5.  [**Whitepaper**](whitepaper.md): Read our in-depth blog post on the philosophy and engineering behind the project.
6.  [**API Reference**](api.md): Detailed documentation for the `ice_lite.sdk`.
7.  [**Project Roadmap**](roadmap.md): The phased engineering and product strategy for GLACIER.

## Getting Help
If you encounter any issues or have questions, please open an issue on the GitHub repository.

## Upgrading to Enterprise
GLACIER and ICE-Lite are designed for local prototypes and single-agent workflows. For multi-tenant isolation (RLS), Redis hot-caching, and hardware-bound security, consider upgrading to **ICE Enterprise**. Read the [Upgrade Guide](https://github.com/Dopove/ICE/blob/main/docs/ICE_vs_ICE-LITE.md) for more details.
