# GLACIER: Mamba with Infinite Memory

**GLACIER** is an open-source project by **Dopove Private Limited** that gives State Space Models like Mamba a persistent, time-aware memory. It solves context rot by integrating a lightweight virtual memory engine (`ICE-Lite`) and a temporal reranking layer (`Temporal-RAG`), ensuring models retain long-term memory while remaining fast and efficient.

This project was architected and built by **Saran S**, Founder & CEO of Dopove.

- **GitHub:** [https://github.com/Saran-386](https://github.com/Saran-386)
- **Dopove:** [https://www.dopove.com/](https://www.dopove.com/)

---

## Key Features

*   **Persistent Memory:** Sessions can be stopped and restarted days or weeks later, with full memory recall.
*   **Temporal-Aware RAG:** A post-retrieval layer that scores and ranks memories based on freshness and validity, preventing the model from using stale information.
*   **Agentic Capabilities:** Native support for multi-step tool use (MCP), long-form generation ("Turbo-Stitching"), and self-improvement dataset creation.
*   **$O(1)$ Inference Speed:** Retains the core speed advantage of Mamba by managing memory externally.

## Getting Started

Full installation and usage instructions are available in the documentation.
> **[Read the Full Documentation](./docs/index.md)**

**Quick-start:**
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download embedding models (see docs/getting_started.md)

# 3. Run the benchmark demo
python -m ice_lite.integration
```

## Project Structure
```
/
├── ice_lite/             # The core ICE-Lite Python package
├── mamba_ssm/            # The Mamba2 source code
├── temporal_rag.py       # The Temporal-RAG source
├── integration.py        # The main GLACIER demo script
├── test_glacier.py       # The full benchmark test suite
├── docs/                 # All project documentation
├── app.py                # Gradio demo for HuggingFace Spaces
├── setup.py              # Packaging script
├── requirements.txt
└── LICENSE               # Main project license (Apache 2.0)
```

## License

This project is licensed under the Apache License 2.0. See the `LICENSE` file for details. Note that sub-components (`mamba_ssm`, `temporal_rag`) are distributed under their own original licenses, which are included in their respective directories.
