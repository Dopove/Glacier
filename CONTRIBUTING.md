# Contributing to GLACIER

We welcome contributions to the GLACIER project! By participating, you agree to abide by our Code of Conduct.

## Code of Conduct

Please review our [Code of Conduct](CODE_OF_CONDUCT.md).

## Development Environment Setup

Getting the development environment right is crucial, especially for compiling the CUDA extensions for Mamba.

1.  **Fork the repository** on GitHub and clone it locally.
2.  **Create and activate a virtual environment.** We strongly recommend using Python 3.12.
    ```bash
    python3.12 -m venv .venv
    source .venv/bin/activate
    ```
3.  **Install dependencies.** Use the detailed installation guide, which handles the specific build requirements for `mamba_ssm`. The `--no-build-isolation` flag is critical.
    ```bash
    # Follow the verified installation steps
    pip install --upgrade pip
    pip install wheel ninja packaging
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install causal-conv1d mamba_ssm --no-build-isolation
    pip install -r requirements.txt
    ```
4.  **Download embedding models.** Follow the instructions in the [Getting Started guide](./docs/getting_started.md) to download the local ONNX models for ICE-Lite.

## Development Workflow

1.  **Create a new branch** for your feature or bug fix: `git checkout -b feature/your-feature-name`
2.  **Make your changes.** Ensure your code adheres to the project's coding style and conventions.
3.  **Write tests.** For new features, add unit and integration tests. For bug fixes, add a test that reproduces the bug.
4.  **Run tests.** Ensure all existing tests pass.
    ```bash
    # Run core memory benchmarks
    python test_glacier.py

    # Run agentic feature tests
    pytest test_agentic_features.py
    ```
5.  **Update documentation.** If your changes affect how GLACIER is used, update the relevant documentation.
6.  **Commit your changes** and open a Pull Request to the `main` branch.

## Contribution Guidelines

*   **Small, focused commits:** Each commit should ideally address a single logical change.
*   **Clear commit messages:** Explain what the commit does and why.
*   **Test coverage:** Aim for high test coverage for your changes.
*   **Respect existing patterns:** Follow the architectural and coding patterns already present in the codebase.

## Licensing

By contributing to GLACIER, you agree that your contributions will be licensed under the Apache License 2.0.
