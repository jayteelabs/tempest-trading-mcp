# Contributing to tempest-trading-mcp

Thank you for your interest in contributing! This document outlines the development workflow, code standards, and processes for contributing to the project.

## Development Setup

### Prerequisites

- Python 3.10, 3.11, or 3.12
- [uv](https://github.com/astral-sh/uv) — fast Python package manager
- **TA-Lib Python 0.6.8+** — install from PyPI; supported wheels bundle the underlying TA-Lib C library on standard CI/dev platforms

### Installing ta-lib

Use the upstream-supported TA-Lib Python wheel install for this repo's NumPy 2 environment:

```bash
uv pip install "TA-Lib==0.6.8"
```

TA-Lib Python 0.6.5+ publishes wheels that bundle the underlying TA-Lib C library, so a separate manual source download/build is not needed on supported platforms.

If your platform does not have a compatible wheel, follow the upstream installation directions instead of using the legacy SourceForge tarball path in this repository:

- https://github.com/ta-lib/ta-lib-python?tab=readme-ov-file#installation

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/jayteelabs/tempest-trading-mcp.git
cd tempest-trading-mcp

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install with dev dependencies
uv venv
source .venv/bin/activate
uv pip install "TA-Lib==0.6.8"
uv pip install -e ".[dev]"

# Verify installation
uv run pytest --collect-only
```

## Code Style

### Formatter and Linter

This project uses [ruff](https://github.com/astral-sh/ruff) for both formatting and linting:

```bash
# Format code
uv run ruff format src/ tests/

# Lint code
uv run ruff check src/ tests/

# Lint with auto-fix
uv run ruff check --fix src/ tests/
```

### Configuration

Ruff is configured in `pyproject.toml`:

- `line-length = 100`
- `target-version = "py310"`
- First-party imports are configured to recognize the `tempest_mcp` package

### Style Rules

- **Type annotations are required** for all public function signatures
- Follow existing import ordering (stdlib → third-party → first-party)
- Use `known-first-party = ["tempest_mcp"]` in ruff to avoid false positives on first-party imports
- Keep lines to 100 characters or fewer

## Branch Strategy

Branches are named to align with Linear ticket IDs:

| Prefix | Example | Purpose |
|--------|---------|---------|
| `feature/eng-XX` | `feature/ENG-48/add-rsi-tool` | New features |
| `fix/eng-XX` | `fix/ENG-55/correct-vwap-calc` | Bug fixes |
| `docs/eng-XX` | `docs/ENG-48/contributing-docs` | Documentation |

Where `XX` is the Linear ticket number (e.g., `ENG-48`).

## Commit Message Convention

This project follows [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <description>

[optional body]
```

### Types

| Type | Use for |
|------|---------|
| `feat:` | New feature or tool |
| `fix:` | Bug fix |
| `docs:` | Documentation only changes |
| `test:` | Adding or updating tests |
| `refactor:` | Code change that neither fixes a bug nor adds a feature |
| `chore:` | Maintenance tasks (dependencies, build config) |
| `perf:` | Performance improvements |

### Examples

```bash
git commit -m "feat: add backtest_elliot_wave tool"
git commit -m "fix: correct RSI calculation for sidelined candles"
git commit -m "docs: add CONTRIBUTING.md and CODE_OF_CONDUCT.md"
git commit -m "test: add integration tests for screener tools"
```

## Pull Request Process

### Before Submitting

1. **Link the Linear ticket** in the PR description (e.g., "Closes ENG-48")
2. **Run the test suite** to ensure all tests pass:

   ```bash
   uv run pytest tests/ -v --tb=short
   ```

3. **Run linting** to ensure code is clean:

   ```bash
   uv run ruff check src/ tests/
   uv run ruff format --check src/ tests/
   ```


### PR Description

Use the [PR template](./.github/pull_request_template.md) to structure your description:

- **Description**: Summary of changes
- **Type of change**: `feat` / `fix` / `docs` / `test` / `refactor`
- **Testing performed**: How you tested the changes
- **Checklist**: Confirm tests pass, lint is clean, ticket is linked

### Review Requirements

- **Required reviewer**: Josh (or designated maintainer)
- All comments must be resolved before merge
- Self-hosted review workflows are used for this repository

## Testing Notes

### Running Tests

```bash
# All tests
uv run pytest tests/ -v --tb=short

# Specific test file
uv run pytest tests/test_backtest_rsi.py -v --tb=short

# Integration tests (require network access and --run-integration flag)
uv run pytest tests/ -v --tb=short -m integration --run-integration
```

### Test Markers

Tests are marked with pytest markers in `pyproject.toml`:

| Marker | Purpose |
|--------|---------|
| `integration` | Tests requiring network access (marked in `conftest.py`) |

### Python Version Matrix

The project supports and tests against:

- Python 3.10
- Python 3.11
- Python 3.12

When adding new features, ensure compatibility across all three versions.

## Issue Reporting

### Bug Reports

When reporting bugs, include:

- Reference the Linear ticket ID when available (e.g., "Related to ENG-48")
- Steps to reproduce the issue
- Expected vs. actual behavior
- Python version and OS

### Feature Requests

Feature requests should be submitted through Linear with the appropriate project label. Before submitting:

1. Search existing tickets to avoid duplicates
2. Consider whether the feature fits the project scope (market data, analytics, backtesting)
3. Provide clear use cases and acceptance criteria

## Project Scope

tempest-trading-mcp provides market data and analytics via the Model Context Protocol:

- **In scope**: Technical indicators, backtesting, screening, sentiment analysis
- **Out of scope**: Trading bots, order execution, position management

## Questions?

For questions about contributing, reach out via:

- GitHub Issues: https://github.com/jayteelabs/tempest-trading-mcp/issues
- Email: tempest@jaytee.cloud
