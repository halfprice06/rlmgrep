# rlmgrep

Grep-shaped CLI search tool for coding agents. The query is interpreted by DSPy RLM; output is rendered in grep-like format.

This tool does **not** provide a classic literal/regex grep mode. It is designed for semantic, RLM-driven search with grep-shaped inputs/outputs.

## Install

```bash
pip install -e .
```

Using uv (tool install):

```bash
uv tool install .
```

## Usage

```bash
rlmgrep [OPTIONS] PATTERN [PATH ...]
```

Examples:

```bash
rlmgrep "where is timeout configured" .
rlmgrep -n -C 2 "token parsing" src/ config/
rlmgrep -g "*.py" --type py "auth token parsing" .
rlmgrep --init-config
```

## Notes

- Directories are searched recursively.
- Use `--no-recursive` to ignore directories and only search explicit files.
- All compatible files are loaded into the RLM directory mapping.
- Incompatible/failed files are skipped with warnings on stderr.
- Output is grep-shaped: line-oriented, with optional `-n` and context flags.
- An ASCII file map is always passed to the RLM for orientation.
- For PDFs, rlmgrep converts to text with page markers and appends `page=N` to output lines.
- Non-text formats (docx/pptx/xlsx/etc.) are converted with MarkItDown.
- Image/audio inputs are only converted when enabled in `~/.rlmgrep`; otherwise they are skipped.
- Image/audio conversions are cached as sidecar files next to the source (e.g. `image.png.md`).
- Use `-g/--glob` and `--type` to filter input files before ingestion.

## RLM Configuration

`rlmgrep` uses DSPy RLM. Configure your model with `--model` or `~/.rlmgrep`.
If `~/.rlmgrep` is missing, a default config is written on first run.

API keys can be provided via `~/.rlmgrep` or the shell environment:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`

### Default Models

- Main model: `openai/gpt-5.2`
- Recursive sub-model: `openai/gpt-5-mini`

### Permanent Config

Store permanent settings in `~/.rlmgrep` (TOML). Example:

```toml
model = "openai/gpt-5.2"
sub_model = "openai/gpt-5-mini"
api_base = "https://api.openai.com/v1"
model_type = "responses"
temperature = 1.0
max_tokens = 64000
max_iterations = 10
max_llm_calls = 20
# Optional: enable image/audio conversion via MarkItDown
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
# api_key = "..." # optional; consider env vars instead
```

## Exit Codes

- `0` when at least one verified match line is printed
- `1` when no matches are printed (no error)
- `2` on fatal errors (e.g. no readable inputs, RLM failure)
