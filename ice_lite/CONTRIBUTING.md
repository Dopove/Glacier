# Contributing to GLACIER

We welcome contributions to the GLACIER project! By participating, you agree to abide by our Code of Conduct.

## Code of Conduct

Please review our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally: `git clone https://github.your-username/glacier.git`
3.  **Create a new branch** for your feature or bug fix: `git checkout -b feature/your-feature-name`
4.  **Install dependencies:** `pip install -r requirements.txt`

## Development Workflow

1.  **Make your changes.** Ensure your code adheres to the project's coding style and conventions.
2.  **Write tests.** For new features, add unit and integration tests. For bug fixes, add a test that reproduces the bug.
3.  **Run tests.** Ensure all existing tests pass: `python -m ice_lite.test_glacier`
4.  **Update documentation.** If your changes affect how GLACIER is used, update the relevant documentation.
5.  **Commit your changes.** Use clear and concise commit messages.
6.  **Push your branch** to your fork: `git push origin feature/your-feature-name`
7.  **Open a Pull Request** (PR) to the `main` branch of the upstream repository.

## Contribution Guidelines

*   **Small, focused commits:** Each commit should ideally address a single logical change.
*   **Clear commit messages:** Explain what the commit does and why.
*   **Test coverage:** Aim for high test coverage for your changes.
*   **Respect existing patterns:** Follow the architectural and coding patterns already present in the codebase.
*   **Feature Parity with Enterprise ICE (where applicable):** When contributing to core memory management components, consider alignment with the design principles of the proprietary ICE Enterprise where applicable, ensuring the upgrade path remains smooth.

## Licensing

By contributing to GLACIER, you agree that your contributions will be licensed under the Apache License 2.0.
