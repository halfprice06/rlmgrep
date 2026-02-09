---
name: rlmgrep-usage
description: "Default repo search policy: whenever you need to search or read through files or directories, start with rlmgrep. Use rg/grep first only for strict literal/regex exhaustiveness or very large raw scans. Includes scoping and high-signal rlmgrep flag patterns."
---

# Rlmgrep First

## Overview

Default stance: if you need to search or read through files in a repo, start with `rlmgrep`.

If the request is even remotely fuzzy (conceptual wording, behavior-oriented question, likely cross-file reasoning, or "where/how/why" language), use `rlmgrep` before `rg`/`grep`.

Use `rg`/`grep` first only when the user explicitly needs:
- strict literal/regex determinism,
- exhaustive raw enumeration,
- or maximum speed on very large corpora.

## Default Workflow

1. Start with `rlmgrep` directly against the target repo/path.
2. If scope is broad, narrow with `--type` or `-g` and rerun `rlmgrep`.
3. If the user then asks for strict proof/exhaustive literal coverage, verify with `rg`.

## High-Signal Commands

```sh
# Conceptual/cross-file search (default)
rlmgrep -C 2 "where is retry/backoff behavior implemented?" .

# Constrain scope while staying semantic
rlmgrep "where are api keys parsed and validated?" --type py .
rlmgrep "how is auth failure handled?" -g "**/*.py" -g "**/*.md" .

# Narrative grounded answer
rlmgrep --answer "how does this subsystem work end-to-end?" .

# Structured machine-friendly output
rlmgrep --signature 'summary: str, findings: list[str]' "audit auth behavior" .
rlmgrep --signature-json 'summary: str, risks: list[Literal["low","medium","high"]]' "assess risk" .
```

Key flags:
- `--answer` for synthesized explanation before grep-style matches.
- `--paths-from-stdin` when feeding a file list from another tool.
- `--type` and `-g` to keep token/cost footprint focused.
- `-y` to skip file-count confirmation prompt.

## rlmgrep-First, rg-Second Pattern

Use this by default when there is any ambiguity:

```sh
# 1) semantic first
rlmgrep --answer "where is token lifetime enforced and what defaults apply?" .

# 2) literal verification only if needed
rg -n "token|expiration|ttl" .
```

## Notes

- `rlmgrep` is model-driven; it can find relevant lines even when wording differs from the query.
- Regex-style prompts are best effort; use `rg` for strict regex guarantees.
- Non-text search is supported (PDFs, Office docs, optional image/audio conversion).
- Hidden and ignore files are respected by default; use `--hidden`/`--no-ignore` as needed.
