# rlmgrep

Grep-shaped search powered by DSPy RLM. It accepts a natural-language query, scans the files you point at, and prints matching lines in a grep-like format.

## Quickstart

```sh
uv tool install --python 3.11 .
# or from GitHub:
# uv tool install --python 3.11 git+https://github.com/halfprice06/rlmgrep.git

export OPENAI_API_KEY=...  # or set keys in ~/.rlmgrep
rlmgrep "where are API keys read" rlmgrep/
```

## Requirements

- Python 3.11+
- Deno runtime (DSPy RLM uses a Deno-based interpreter)
- API key for your chosen provider (OpenAI, Anthropic, Gemini, etc.)

## Install Deno

DSPy requires the Deno runtime. Install it with the official scripts:

macOS/Linux:

```sh
curl -fsSL https://deno.land/install.sh | sh
```

Windows PowerShell:

```powershell
irm https://deno.land/install.ps1 | iex
```

Verify it is on your `PATH`:

```sh
deno --version
```

## Usage

```sh
rlmgrep [options] "query" [paths...]
```

Common options:

- `-n` show line numbers
- `-H` always show filenames
- `-C N` context lines before/after (grep-style)
- `-A N` context lines after
- `-B N` context lines before
- `-m N` max matching lines per file
- `-g GLOB` include files matching glob (repeatable, comma-separated)
- `--type T` include file types (repeatable, comma-separated)
- `--no-recursive` do not recurse directories
- `-a`, `--text` treat binary files as text
- `--model`, `--sub-model` override model names
- `--api-key`, `--api-base`, `--model-type` override provider settings
- `--max-iterations`, `--max-llm-calls` cap RLM search effort
- `-v`, `--verbose` show verbose RLM output

Examples:

```sh
# Natural-language query over a repo
rlmgrep -n -C 2 "token parsing" rlmgrep/

# Restrict to Python files
rlmgrep "where config is read" --type py rlmgrep/

# Glob filters (repeatable or comma-separated)
rlmgrep "error handling" -g "**/*.py" -g "**/*.md" .

# Read from stdin (only when no paths are provided)
cat README.md | rlmgrep "install"
```

## Input selection

- Directories are searched recursively by default. Use `--no-recursive` to stop recursion.
- `--type` uses built-in type mappings (e.g., `py`, `js`, `md`); unknown values are treated as file extensions.
- `-g/--glob` matches path globs against normalized paths (forward slashes).
- Paths are printed relative to the current working directory when possible.
- If no paths are provided, rlmgrep reads from stdin and uses the synthetic path `<stdin>`; if stdin is empty, it exits with code 2.

## Output contract (stable for agents)

- Matches are written to stdout; warnings go to stderr.
- Output uses grep-like prefixes:
  - `path:line:text` for match lines when both `-H` and `-n` are enabled.
  - `path-line-text` for context lines (note the `-` separator).
  - If `-H` or `-n` are omitted, their parts are omitted.
- Line numbers are 1-based.
- When context ranges are disjoint, a `--` line separates groups.
- Exit codes:
  - `0` = at least one match
  - `1` = no matches
  - `2` = usage/config/error

Agent tip: use `-n -H` and no context for parse-friendly output, then key off exit codes.

## Configuration

rlmgrep creates a default config automatically if missing. The config path is:

- `~/.rlmgrep/config.toml`

Default config values (from `rlmgrep/config.py`):

```toml
model = "openai/gpt-5.2"
sub_model = "openai/gpt-5-mini"
api_base = "https://api.openai.com/v1"
model_type = "responses"
temperature = 1.0
max_tokens = 64000
max_iterations = 10
max_llm_calls = 20
# markitdown_enable_images = false
# markitdown_image_llm_model = "gpt-5-mini"
# markitdown_image_llm_provider = "openai"
# markitdown_image_llm_api_key = ""
# markitdown_image_llm_api_base = ""
# markitdown_image_llm_prompt = ""
# markitdown_enable_audio = false
# markitdown_audio_model = "gpt-4o-mini-transcribe"
# markitdown_audio_provider = "openai"
# markitdown_audio_api_key = ""
# markitdown_audio_api_base = ""
```

CLI flags override config values. Model keys are resolved as:

1) CLI flags (`--api-key`, `--sub-api-key`)
2) Config values (`api_key`, `sub_api_key`)
3) Provider env vars inferred from the model name:
   - `OPENAI_API_KEY`
   - `ANTHROPIC_API_KEY`
   - `GEMINI_API_KEY`

If more than one provider key is set and the model does not make the provider obvious, rlmgrep emits a warning and requires an explicit `--api-key`.

## Non-text files (PDF, images, audio)

- PDF files are parsed with `pypdf`. Each page gets a marker line `===== Page N =====`, and output lines include a `page=N` suffix.
- Images and audio are converted via `markitdown` when enabled in config. For image/audio conversion, an `openai` Python client is required.
- Converted image/audio text is cached in sidecar files named `<original>.<ext>.md` next to the original file and reused on subsequent runs.
- Use `-a/--text` to force binary files to be read as text (UTF-8 with replacement).

## Agent usage notes

- Prefer narrow corpora (globs/types) to reduce token usage.
- Use `--max-llm-calls` to cap costs; combine with small `--max-iterations` for safety.
- Always read stderr for warnings (skipped files, config issues, ambiguous API keys).
- For reproducible parsing, use `-n -H` and avoid context (`-C/-A/-B`).
- RLM results are verified against real file lines; invalid or duplicate matches are dropped and reported.

## Development

- Install locally: `pip install -e .` or `uv tool install .`
- Run: `rlmgrep "query" .`
- No test suite is configured yet.

## Security

Do not commit API keys. Use environment variables or `~/.rlmgrep/config.toml`.
