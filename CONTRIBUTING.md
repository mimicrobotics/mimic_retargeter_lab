# Contributing to mimic_retargeter_lab

Thanks for your interest in contributing. This document covers the development workflow. For environment setup, dependencies, and GPU configuration, follow the [README](README.md) — particularly the [Installation](README.md#installation) and [Setup](README.md#setup) sections, which cover the uv-based install and the GPU requirements for the sampling-based retargeter.

## Development install

Once your environment is set up per the README, install in editable mode:

```bash
uv sync
```

## Pre-commit hooks

The repo uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting, enforced via pre-commit. Enable the hooks once after cloning:

```bash
uv run pre-commit install
```

Commits will then run ruff automatically on staged files. To run the full suite manually:

```bash
uv run pre-commit run --all-files
```

## Running tests

```bash
uv run pytest tests/
```

The test suite covers hand model kinematics and retargeter correctness via golden `.npz` fixtures. Tests do not require GPU or external datasets.

If you add a new robot hand, generate its golden data first:

```bash
uv run python scripts/generate_robot_hand_golden_data.py hand=<hand_name>
```

## Submitting changes

1. Open an issue describing the bug or feature before starting large changes.
2. Work on a branch off `main`.
3. Ensure `pre-commit run --all-files` and `pytest tests/` both pass.
4. Open a pull request with a clear description of what changed and why.

## License

By contributing you agree that your contributions will be licensed under the same [CC BY-NC 4.0](LICENSE) license as the rest of the project.
