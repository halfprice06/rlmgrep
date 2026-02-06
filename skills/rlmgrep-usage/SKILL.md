---
name: rlmgrep-usage
description: Decide when to use grep/rg (literal, fast, exhaustive) vs rlmgrep (semantic, slower, cross-file reasoning), especially for conceptual questions, cross-file flows, and mixed-format corpora (PDFs/images/audio). Covers scoping strategies, regex-in-natural-language best-effort behavior, and key flags (`--paths-from-stdin`, `--answer`, `-g`, `--type`, `-y`) that make rlmgrep reliable and cost-aware.
---

# Rlmgrep Usage

## Overview

Use this skill to pick the right search tool and craft effective rlmgrep usage. It explains when to stick to grep/rg for literal matches and when to use rlmgrep for semantic, multi-file, or non-text searches.

## Decision Guide

Use `grep`/`rg` when:
- You need exact literal or regex matches.
- You need deterministic, exhaustive results.
- You are searching huge repos and want raw speed.
- The query is a single identifier, string, or known regex.

Use `rlmgrep` when:
- The query is natural language or conceptual (e.g., “where is retry logic defined?”).
- You need semantic matches that do not necessarily contain the literal query text.
- You want the tool to reason across files and return relevant line numbers.
- You want to include PDFs or other non-text files (when configured).
- You need a “what/where/how” answer grounded in code and docs, not just raw matches.

## rlmgrep Quick Use

Examples:

```sh
rlmgrep -C 2 "where is api key parsed" .
rlmgrep "retry logic for 429" --type py .
rlmgrep "find config defaults" -g "**/*.toml" -g "**/*.py" .
rg -l "token" . | rlmgrep --paths-from-stdin --answer "what does this token control?"
rlmgrep --signature 'summary: str, findings: list[str]' "summarize key behaviors" .
rlmgrep --signature-json 'summary: str, levels: list[Literal["low","medium","high"]]' "summarize risk levels" .
```

Notes:
- rlmgrep is slower and may cost tokens; use it when semantic reasoning is worth it.
- rlmgrep is grep-shaped output, but the match semantics are model-driven.
- `--paths-from-stdin` treats stdin as newline-delimited file paths. Without it, piped stdin is treated as file content.
- Hidden files and ignore files are respected by default. Use `--hidden` to include dotfiles, and `--no-ignore` to bypass `.gitignore`/`.rgignore`/`.ignore` (similar to rg).
- rlmgrep asks for confirmation when more than 1000 files would be loaded (use `-y/--yes` to skip). It aborts over 5000 files by default.
  - `-y/--yes` only skips the confirmation prompt; it does not bypass the 5000-file abort cap.
  - To run on larger repos, pre-filter with `rg -l ...`, or use `-g/--type` to narrow the file set.
- You can include regex-style patterns inside a natural-language prompt, and the RLM may use Python `re` internally to approximate that logic, but results are not guaranteed to match `grep`/`rg` exactly.
- Non-text inputs: PDFs are parsed, images can be described via LLMs (OpenAI/Anthropic/Gemini), and audio transcription is OpenAI-only.
- MarkItDown is loaded lazily. Text-only runs skip it; non-text conversion is enabled only when needed.
- OpenRouter: set `model` to `openrouter/...`, set `api_base` to `https://openrouter.ai/api/v1`, and provide `OPENROUTER_API_KEY` (or `api_key` in config).

## Structured Output Signatures

Use this when the caller wants machine-friendly output or custom report fields.

- `--signature '<fields>'` prints sectioned markdown-like text for humans.
- `--signature-json '<fields>'` prints one compact JSON object to stdout for piping/saving.
- Provide output fields only (for example: `summary: str, findings: list[str]`). Do not include inputs or `->`.
- Supported output types: `str`, `int`, `float`, `bool`, `list[T]`, `dict[str, T]`, `Literal[...]`.
- JSON mapping: `str` -> string, `int/float` -> number, `bool` -> boolean, `list[T]` -> array, `dict[str, T]` -> object, `Literal[...]` -> scalar.
- In `--signature-json` mode, progress/warnings still go to stderr.

Example (best-effort regex semantics + extra context):

```sh
rlmgrep "Find JavaScript files with functions matching `function\\s+use[A-Z]\\w+` and mention hooks." .
```

## Optional Workflow: rg → rlmgrep

If you already know a literal anchor (e.g., a class name, constant, or table), use `rg` to find the files quickly, then pipe those file paths into rlmgrep for semantic reasoning.

Examples:

```sh
# Find candidate files quickly, then ask a semantic question across them
rg -l "auth" . | rlmgrep --paths-from-stdin --answer "where is auth handled?"

# Use a literal filter to narrow the corpus, then ask for explanation
rg -l "retry" . | rlmgrep --paths-from-stdin --answer "how do retries work?"

# Restrict to a file type first, then ask a conceptual question
rg -l "token" --type py . | rlmgrep --paths-from-stdin --answer "where are tokens parsed?"
```

## Default Tool Choice Pattern

If you can express the search as a literal token or regex, start with `rg` for speed and exhaustiveness. If that fails, or the question is semantic, switch to `rlmgrep` with a short natural-language prompt.

## When Not To Use rlmgrep

Use `rg` or `grep` if you need guaranteed literal matches, very large-scale scanning, or exact regex behavior. rlmgrep does not promise literal containment.
