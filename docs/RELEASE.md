# Release Process

This document describes the release workflow for `tempest-tradingview-mcp` and what happens at each gate.

## Release Gates

This repository handles **release preparation** (ENG-51 scope). The actual publication steps require additional privileged actions documented below.

### What's Done Here (Release Prep)

- [x] Version bumped to `1.0.0` in `pyproject.toml`, `src/tempest_mcp/__init__.py`, and `src/tempest_mcp/config.py`
- [x] `CHANGELOG.md` created with shipped capability summary
- [x] `.github/workflows/release.yml` added for build/publish automation
- [x] Package status promoted from Alpha to Beta in `pyproject.toml`

### What Happens at the Release Gate (Not in this ticket)

The following privileged actions are **not** performed by this ticket and require manual gate approval:

1. **Git tag creation** — `git tag v1.0.0` with signed commit
2. **GitHub Release draft** — Draft release notes from `CHANGELOG.md`
3. **TestPyPI publish** — Dry-run validation against TestPyPI
4. **PyPI publish** — Live publication to PyPI
5. **GitHub environment/PyPI secret setup** — `PYPI_API_TOKEN` and `TESTPYPI_API_TOKEN` in GitHub secrets

## Triggering a Release

### Prerequisites

1. All tests pass on `main`
2. `CHANGELOG.md` is updated with the target version
3. Version is bumped in all three surfaces:
   - `pyproject.toml` (`version`)
   - `src/tempest_mcp/__init__.py` (`__version__`)
   - `src/tempest_mcp/config.py` (`mcp_server_version` default + `_get_str` default)
4. GitHub secrets configured:
   - `PYPI_API_TOKEN` — for live PyPI publication
   - `TESTPYPI_API_TOKEN` — for TestPyPI validation

### Workflow Dispatch (Manual Trigger)

Use manual dispatch for release-build validation. An approved release-gate run that publishes to TestPyPI must be launched from `main`.

To trigger a release build manually:

```bash
# Navigate to Actions > Release Package > Run workflow
# Or use GitHub CLI:
gh workflow run release.yml --ref main -f dry_run=false
```

Inputs:
- `dry_run` (optional, default: `true`) — Set to `false` to publish the already-versioned build to TestPyPI from `main`

Manual dispatch behavior:

1. The workflow builds the wheel and sdist artifacts
2. It validates the built package with a local install check
3. If `dry_run=false` **and** the workflow is dispatched from `main`, it publishes the artifacts to TestPyPI

### Tag Push Publish (Release Gate)

Production publication is gated behind an explicit release tag push.

When an approved `v*.*.*` tag is pushed:

1. The `release.yml` workflow detects the tag push
2. It builds the package artifacts
3. It validates the build locally
4. If credentials are available, it publishes to PyPI

**Note:** This tag-gated publish path requires the `PYPI_API_TOKEN` secret to be configured in the repository.

## Release Artifact Flow

```
workflow_dispatch
       │
       ├──► Build wheel + sdist
       │
       ├──► Validate artifacts (test install)
       │
       └──► (if dry_run=false on main) Publish to TestPyPI

push tag v*.*.*
       │
       ├──► Build wheel + sdist
       │
       ├──► Validate artifacts (test install)
       │
       └──► Publish to PyPI
```

## Version Alignment

All three version surfaces must be updated together:

| File | Field | Example |
|------|-------|---------|
| `pyproject.toml` | `[project].version` | `1.0.0` |
| `src/tempest_mcp/__init__.py` | `__version__` | `"1.0.0"` |
| `src/tempest_mcp/config.py` | `Config.mcp_server_version` default + `_get_str("MCP_SERVER_VERSION", "1.0.0")` | `"1.0.0"` |

## PyPI/Twine Configuration

The release workflow uses `pypa/gh-action-pypi-publish` for PyPI publication. The `.pypirc` file is not required since the action handles authentication.

For local testing:

```bash
# Install twine if needed
pip install twine

# Upload to TestPyPI first (dry run)
python -m twine upload --repository testpypi dist/*

# Upload to PyPI (production)
python -m twine upload dist/*
```

## Changelog Maintenance

Before each release:

1. Add a new `[X.Y.Z] - YYYY-MM-DD` section at the top of `CHANGELOG.md`
2. Move unreleased changes into the new version section
3. Group changes by type: Added, Changed, Deprecated, Removed, Fixed, Security
4. Reference specific capability areas rather than individual ticket numbers

## Rollback Procedure

If a bad release is published to PyPI:

1. **Do not delete the release from PyPI** — PyPI does not support deletion
2. Yank the bad version: `gh release edit vX.Y.Z --prerelease` or contact PyPI support
3. Bump to next patch version in a new commit
4. Publish the corrected version

## Support

For release issues, contact the Tempest Engineering team.
