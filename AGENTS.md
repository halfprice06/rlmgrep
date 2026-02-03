# Repository Guidelines

## Project Structure & Module Organization
- `rlmgrep/` contains the CLI implementation.
  - `cli.py` is the entry point for argument parsing and orchestration.
  - `ingest.py`, `render.py`, and `interpreter.py` handle file loading, output shaping, and query execution.
  - `config.py` and `rlm.py` manage model configuration and DSPy integration.
- `rlmgrep.egg-info/` is generated build metadata; do not edit by hand.
- `README.md` documents install, usage, and configuration.

## Build, Test, and Development Commands
- Install in editable mode: `pip install -e .` (fast local iteration).
- Alternative tool install: `uv tool install .` (local path) or `uv tool install --python 3.11 rlmgrep` (PyPI).
- Run locally: `rlmgrep "query" .` (example: `rlmgrep -n -C 2 "token parsing" rlmgrep/`).
- No automated test command is configured yet.

## Packaging & PyPI Release
- Version lives in `pyproject.toml`. Bump before publishing.
- Build: `python -m build`
- Publish: `python -m twine upload dist/*`
- Ensure `README.md` reflects the latest CLI/config behavior before publish.


## Coding Style & Naming Conventions
- Python only; use 4-space indentation and follow PEP 8.
- Use `snake_case` for functions/variables and `CapWords` for classes.
- Prefer small, single-purpose functions to keep the CLI flow readable.
- No formatter or linter is configured; keep diffs tidy and consistent with existing files.

## Testing Guidelines
- The repository currently has no test suite or `tests/` directory.
- If you add tests, place them under `tests/` and document the runner in `README.md` and `pyproject.toml`.

## Commit & Pull Request Guidelines
- Git history is not present in this checkout, so no commit convention can be inferred.
- Use clear, imperative commit messages (e.g., `Add PDF page markers`).
- PRs should include: a short summary, rationale, and a CLI example that demonstrates the change.

## Security & Configuration Tips
- API keys should be supplied via environment variables (e.g., `OPENAI_API_KEY`) or `~/.rlmgrep`.
- Never commit local config files or secrets.
