# Stack Context

Generated: 2026-04-13

## Stack
- **Language**: Python 3.10+
- **Framework**: MCP server package, pandas/numpy data layer
- **Build**: `hatchling`, `uv`
- **Test**: `pytest tests/`
- **Lint**: `ruff check src/ tests/` (CI gate: yes)
- **Format**: Ruff-managed import/style rules (CI gate: partial via lint)

## Secondary Languages
- YAML (GitHub Actions CI/workflow automation)

## Conventions
- Error handling: custom exception hierarchy plus empty-result fallback in adapters
- Module structure: `src/tempest_mcp/` package with `data/`, `tools/`, `models/`, `indicators/`
- Naming: snake_case modules/functions, dataclasses in `models/`
- Tests: `tests/` with focused adapter/indicator/backtest modules; some tests marked `integration`

## CI Gates
- Install dependencies with `uv`
- Lint with `ruff`
- Run `pytest` on Python 3.10, 3.11, and 3.12
