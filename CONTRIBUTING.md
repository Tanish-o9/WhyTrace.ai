# Contributing to whyTrace.ai

Thank you for your interest in contributing to **whyTrace.ai**! We welcome contributions to failure attribution algorithms, new framework integrations, benchmarks, and documentation.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Tanish-o9/WhyTrace.ai.git
   cd WhyTrace.ai
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # On Linux/macOS:
   source .venv/bin/activate
   # On Windows:
   .venv\Scripts\activate
   ```

3. **Install dependencies in editable mode with development tools:**
   ```bash
   pip install -e ".[dev,dashboard]"
   ```

## Running Tests

All unit and integration tests are offline and execute in under 2 seconds:
```bash
pytest -q
```

To run end-to-end regression validation:
```bash
whytrace demo --out runs/dev_demo
whytrace regress --path runs/dev_demo/regressions.jsonl
```

## Pull Request Guidelines

1. **Branch Naming**: Use descriptive prefixes like `feat/`, `fix/`, `docs/`, or `refactor/`.
2. **Conventional Commits**: Write clear commit messages following Conventional Commits format (e.g. `feat(replay): add heuristic counterfactual generator`).
3. **Tests**: Ensure any new features or bug fixes include corresponding unit tests in `tests/`.
4. **Code Style**: Run code formatters and linters before submitting PRs.

