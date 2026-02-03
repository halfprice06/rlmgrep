from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import dspy
from . import __version__
from .config import ensure_default_config, load_config
from .file_map import build_file_map
from .ingest import FileRecord, collect_candidates, load_files, resolve_type_exts
from .rlm import Match, build_lm, run_rlm
from .render import render_matches


def _warn(msg: str) -> None:
    print(f"rlmgrep: {msg}", file=sys.stderr)


def _confirm_over_limit(count: int, threshold: int) -> bool:
    prompt = (
        f"rlmgrep: {count} files to load (over {threshold}). Continue? [y/N] "
    )
    try:
        with open("/dev/tty", "r+") as tty:
            print(prompt, file=tty, end="", flush=True)
            response = tty.readline()
    except Exception:
        if not sys.stdin.isatty():
            _warn("refusing to prompt for confirmation; use --yes to proceed")
            return False
        print(prompt, file=sys.stderr, end="", flush=True)
        response = sys.stdin.readline()
    return response.strip().lower() in {"y", "yes"}


def verify_matches(
    matches: list[Match],
    files: dict[str, FileRecord],
) -> tuple[dict[str, list[int]], int]:
    verified: dict[str, list[int]] = {}
    dropped = 0

    seen: set[tuple[str, int]] = set()
    for match in matches:
        record = files.get(match.path)
        if record is None or not record.lines:
            dropped += 1
            continue

        line_no = match.line
        if not (1 <= line_no <= len(record.lines)):
            dropped += 1
            continue

        key = (match.path, line_no)
        if key in seen:
            continue
        seen.add(key)
        verified.setdefault(match.path, []).append(line_no)

    return verified, dropped


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rlmgrep",
        description="Grep-shaped CLI search powered by DSPy RLM.",
    )
    parser.add_argument("--version", action="version", version=f"rlmgrep {__version__}")
    parser.add_argument("pattern", nargs="?", help="Query string (interpreted by RLM)")
    parser.add_argument("paths", nargs="*", help="Files or directories")

    parser.add_argument("-n", dest="line_numbers", action="store_true", help="Show line numbers")
    parser.add_argument("-H", dest="with_filename", action="store_true", help="Always show filenames")
    parser.add_argument("-r", dest="recursive", action="store_true", help="Recursive (directories are searched recursively by default)")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false", help="Do not recurse directories")
    parser.set_defaults(recursive=True)

    parser.add_argument("-C", dest="context", type=int, default=0, help="Context lines before/after")
    parser.add_argument("-A", dest="after", type=int, default=None, help="Context lines after")
    parser.add_argument("-B", dest="before", type=int, default=None, help="Context lines before")
    parser.add_argument("-m", dest="max_count", type=int, default=None, help="Max matching lines per file")
    parser.add_argument("-a", "--text", dest="binary_as_text", action="store_true", help="Search binary files as text")
    parser.add_argument("--answer", action="store_true", help="Print a narrative answer before grep output")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip file count confirmation")
    parser.add_argument(
        "--stdin-files",
        action="store_true",
        help="Treat stdin as newline-delimited file paths",
    )

    parser.add_argument("-g", "--glob", dest="globs", action="append", default=[], help="Include files matching glob (may repeat)")
    parser.add_argument("--type", dest="types", action="append", default=[], help="Include file types (py, js, md, etc.). May repeat")

    parser.add_argument("--model", type=str, default=None, help="DSPy model name")
    parser.add_argument("--sub-model", type=str, default=None, help="Recursive sub-model name")
    parser.add_argument("--api-base", type=str, default=None, help="Provider API base URL")
    parser.add_argument("--api-key", type=str, default=None, help="Provider API key")
    parser.add_argument("--model-type", type=str, default=None, help="Model type (e.g., chat)")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p sampling")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens for generation")
    parser.add_argument("--sub-api-base", type=str, default=None, help="Sub-model API base URL")
    parser.add_argument("--sub-api-key", type=str, default=None, help="Sub-model API key")
    parser.add_argument("--sub-model-type", type=str, default=None, help="Sub-model type (e.g., chat)")
    parser.add_argument("--sub-temperature", type=float, default=None, help="Sub-model temperature")
    parser.add_argument("--sub-top-p", type=float, default=None, help="Sub-model top-p")
    parser.add_argument("--sub-max-tokens", type=int, default=None, help="Sub-model max tokens")

    parser.add_argument("--max-iterations", type=int, default=None, help="RLM max iterations")
    parser.add_argument("--max-llm-calls", type=int, default=None, help="RLM max LLM calls")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose RLM output")
    parser.add_argument(
        "--init-config",
        action="store_true",
        help="Write default config to ~/.rlmgrep if it does not exist",
    )

    return parser.parse_args(argv)


def _pick(cli_value, config: dict, key: str, default=None):
    if cli_value is not None:
        return cli_value
    if key in config:
        return config[key]
    return default


def _env_value(name: str) -> str | None:
    val = os.getenv(name)
    if val is None:
        return None
    val = val.strip()
    return val or None


def _detect_provider(model: str | None) -> str | None:
    if not model:
        return None
    if "/" in model:
        return model.split("/", 1)[0].lower()
    lower = model.lower()
    if "anthropic" in lower:
        return "anthropic"
    if "gemini" in lower or "google" in lower:
        return "gemini"
    if "openai" in lower or "gpt" in lower:
        return "openai"
    return None


def _pick_api_key(
    cli_value,
    config: dict,
    config_key: str,
    model: str | None,
    warnings: list[str],
    label: str,
) -> str | None:
    if cli_value is not None:
        return cli_value
    if config_key in config:
        return config[config_key]

    provider = _detect_provider(model)
    provider_env = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }

    if provider in provider_env:
        return _env_value(provider_env[provider])

    env_names = list(provider_env.values())
    found = [(name, _env_value(name)) for name in env_names]
    found = [(name, val) for name, val in found if val]
    if len(found) == 1:
        return found[0][1]
    if len(found) > 1:
        warnings.append(
            f"ambiguous API key env vars for {label}; set --api-key or ~/.rlmgrep"
        )
    return None


def _parse_num(value, cast):
    if value is None:
        return None
    try:
        return cast(value)
    except Exception:
        return None


def _split_list(values: list[str]) -> list[str]:
    out: list[str] = []
    for item in values:
        out.extend([part for part in item.split(",") if part])
    return out


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _build_markitdown(config: dict, warnings: list[str]):
    try:
        from markitdown import MarkItDown  # type: ignore
    except Exception:
        warnings.append("markitdown not installed; non-text files may be skipped")
        return None, False, False, None

    enable_images = _as_bool(config.get("markitdown_enable_images"))
    enable_audio = _as_bool(config.get("markitdown_enable_audio"))
    llm_model = config.get("markitdown_image_llm_model")
    llm_provider = (config.get("markitdown_image_llm_provider") or "openai").lower()
    llm_prompt = config.get("markitdown_image_llm_prompt")
    llm_api_key = config.get("markitdown_image_llm_api_key")
    llm_api_base = config.get("markitdown_image_llm_api_base")

    def _openai_client(api_key, api_base, warn: str):
        try:
            from openai import OpenAI  # type: ignore
        except Exception:
            warnings.append(warn)
            return None
        kwargs: dict[str, str] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if api_base:
            kwargs["base_url"] = api_base
        return OpenAI(**kwargs)

    llm_client = None
    if enable_images:
        if not llm_model:
            warnings.append(
                "markitdown_enable_images set but markitdown_image_llm_model missing; skipping images"
            )
            enable_images = False
        elif llm_provider != "openai":
            warnings.append(
                f"markitdown image LLM provider '{llm_provider}' not supported; skipping images"
            )
            enable_images = False
        else:
            llm_client = _openai_client(
                llm_api_key,
                llm_api_base,
                "openai package missing; skipping image conversion",
            )
            if llm_client is None:
                enable_images = False

    md_kwargs: dict[str, object] = {"enable_plugins": False}
    if llm_client and enable_images:
        md_kwargs["llm_client"] = llm_client
        md_kwargs["llm_model"] = llm_model
        if llm_prompt:
            md_kwargs["llm_prompt"] = llm_prompt

    audio_transcriber = None
    audio_model = config.get("markitdown_audio_model")
    audio_provider = (config.get("markitdown_audio_provider") or "openai").lower()
    audio_api_key = config.get("markitdown_audio_api_key")
    audio_api_base = config.get("markitdown_audio_api_base")

    if enable_audio:
        if not audio_model:
            warnings.append(
                "markitdown_enable_audio set but markitdown_audio_model missing; using default audio converter"
            )
        elif audio_provider != "openai":
            warnings.append(
                f"markitdown audio provider '{audio_provider}' not supported; using default audio converter"
            )
        else:
            audio_client = _openai_client(
                audio_api_key,
                audio_api_base,
                "openai package missing; using default audio converter",
            )
            if audio_client:

                def _transcribe_audio(path: Path) -> str:
                    with path.open("rb") as f:
                        resp = audio_client.audio.transcriptions.create(
                            model=audio_model, file=f
                        )
                    return getattr(resp, "text", "") or ""

                audio_transcriber = _transcribe_audio

    return MarkItDown(**md_kwargs), enable_images, enable_audio, audio_transcriber


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    cwd = Path.cwd()

    if args.init_config:
        config_path, err = ensure_default_config()
        if err:
            _warn(f"failed to write config: {err}")
            return 2
        if config_path is None:
            _warn("config already exists")
        else:
            print(str(config_path))
        return 0

    if not args.pattern:
        _warn("missing pattern")
        return 2

    _, err = ensure_default_config()
    if err:
        _warn(f"failed to write default config: {err}")
    config, config_warnings = load_config()
    for w in config_warnings:
        _warn(w)

    # Resolve input corpus.
    globs = _split_list(args.globs)
    type_names = _split_list(args.types)

    type_exts, type_warnings = resolve_type_exts(type_names)
    for w in type_warnings:
        _warn(w)

    md_warnings: list[str] = []
    markitdown, md_enable_images, md_enable_audio, audio_transcriber = _build_markitdown(
        config, md_warnings
    )
    for w in md_warnings:
        _warn(w)

    input_paths: list[str] | None = None
    stdin_text: str | None = None
    if args.paths:
        input_paths = list(args.paths)
    elif args.stdin_files:
        if sys.stdin.isatty():
            _warn("no input paths and stdin is empty")
            return 2
        raw = sys.stdin.read()
        input_paths = [line.strip() for line in raw.splitlines() if line.strip()]
        if not input_paths:
            _warn("stdin contained no file paths")
            return 2
    else:
        if sys.stdin.isatty():
            _warn("no input paths and stdin is empty")
            return 2
        stdin_text = sys.stdin.read()

    if input_paths is None:
        text = stdin_text or ""
        files = {
            "<stdin>": FileRecord(path="<stdin>", text=text, lines=text.split("\n"))
        }
        warnings: list[str] = []
    else:
        warn_threshold = _parse_num(
            _pick(None, config, "file_warn_threshold", 200), int
        )
        hard_max = _parse_num(_pick(None, config, "file_hard_max", 1000), int)
        if warn_threshold is not None and warn_threshold <= 0:
            warn_threshold = None
        if hard_max is not None and hard_max <= 0:
            hard_max = None

        candidates = collect_candidates(
            input_paths,
            cwd=cwd,
            recursive=args.recursive,
            include_globs=globs,
            type_exts=type_exts,
        )
        candidate_count = len(candidates)
        if hard_max is not None and candidate_count > hard_max:
            _warn(
                f"{candidate_count} files to load (over {hard_max}); aborting"
            )
            return 2
        if (
            warn_threshold is not None
            and candidate_count > warn_threshold
            and not args.yes
        ):
            if not _confirm_over_limit(candidate_count, warn_threshold):
                return 2

        files, warnings = load_files(
            candidates,
            cwd=cwd,
            markitdown=markitdown,
            enable_images=md_enable_images,
            enable_audio=md_enable_audio,
            audio_transcriber=audio_transcriber,
            binary_as_text=args.binary_as_text,
        )

    for w in warnings:
        _warn(w)

    if not files:
        _warn("no readable files")
        return 2

    # Resolve model configuration.
    model = _pick(args.model, config, "model", "openai/gpt-5.2")
    sub_model = _pick(args.sub_model, config, "sub_model", "openai/gpt-5-mini")

    warnings: list[str] = []

    api_base = _pick(args.api_base, config, "api_base", None)
    api_key = _pick_api_key(
        args.api_key, config, "api_key", model, warnings, "main model"
    )
    model_type = _pick(args.model_type, config, "model_type", None)
    temperature = _parse_num(_pick(args.temperature, config, "temperature", None), float)
    top_p = _parse_num(_pick(args.top_p, config, "top_p", None), float)
    max_tokens = _parse_num(_pick(args.max_tokens, config, "max_tokens", None), int)

    sub_api_base = _pick(args.sub_api_base, config, "sub_api_base", None)
    sub_api_key = _pick_api_key(
        args.sub_api_key, config, "sub_api_key", sub_model, warnings, "sub-model"
    )
    sub_model_type = _pick(args.sub_model_type, config, "sub_model_type", None)
    sub_temperature = _parse_num(
        _pick(args.sub_temperature, config, "sub_temperature", None), float
    )
    sub_top_p = _parse_num(_pick(args.sub_top_p, config, "sub_top_p", None), float)
    sub_max_tokens = _parse_num(
        _pick(args.sub_max_tokens, config, "sub_max_tokens", None), int
    )

    max_iterations = _parse_num(
        _pick(args.max_iterations, config, "max_iterations", None), int
    )
    max_llm_calls = _parse_num(
        _pick(args.max_llm_calls, config, "max_llm_calls", None), int
    )

    for w in warnings:
        _warn(w)

    # Configure LM.
    try:
        main_lm = build_lm(
            model=model,
            api_base=api_base,
            api_key=api_key,
            model_type=model_type,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        sub_lm = build_lm(
            model=sub_model,
            api_base=sub_api_base or api_base,
            api_key=sub_api_key or api_key,
            model_type=sub_model_type or model_type,
            temperature=sub_temperature if sub_temperature is not None else temperature,
            top_p=sub_top_p if sub_top_p is not None else top_p,
            max_tokens=sub_max_tokens if sub_max_tokens is not None else max_tokens,
        )
    except RuntimeError as exc:
        _warn(str(exc))
        return 2

    dspy.configure(lm=main_lm)

    file_map = build_file_map(sorted(files.keys()))

    directory = {k: v.text for k, v in files.items()}

    try:
        proposed, answer = run_rlm(
            directory=directory,
            query=args.pattern,
            file_map=file_map,
            max_iterations=max_iterations,
            max_llm_calls=max_llm_calls,
            verbose=args.verbose,
            sub_lm=sub_lm,
            with_answer=args.answer,
        )
    except Exception as exc:  # pragma: no cover - defensive
        _warn(f"RLM failure: {exc}")
        return 2

    verified, dropped = verify_matches(proposed, files)
    if dropped:
        _warn(f"dropped {dropped} unverified matches")

    # Apply -m max count per file.
    if args.max_count is not None:
        for path, lines in list(verified.items()):
            limited = sorted(lines)[: args.max_count]
            verified[path] = limited

    # Compute context values.
    before = args.before if args.before is not None else args.context
    after = args.after if args.after is not None else args.context

    output_lines = render_matches(
        files=files,
        matches=verified,
        show_line_numbers=args.line_numbers,
        show_filename=args.with_filename,
        before=before,
        after=after,
    )

    if args.answer:
        if answer:
            print(answer.strip())
        print("--")

    for line in output_lines:
        print(line)

    total_matches = sum(len(lines) for lines in verified.values())
    if total_matches > 0:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
