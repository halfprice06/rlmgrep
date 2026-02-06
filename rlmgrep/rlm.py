from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, get_args, get_origin

import dspy
from pydantic import BaseModel

from .interpreter import RLMGrepInterpreter


class Match(BaseModel):
    path: str
    line: int


class RLMGrepSignature(dspy.Signature):
    """
    You are the search engine for rlmgrep, a grep-shaped CLI for coding agents.
    Inputs include a directory mapping of files (path-to-text), an ASCII file
    map, and a user query string. Your output must be grep-printable matches as
    (path, line) pairs that point to real lines in the provided texts.
    The query may be natural language or a short pattern; interpret it freely to
    find relevant lines. Return all relevant matches you can find, avoid duplicates,
    and only use exact paths from the directory keys.
    Always read the ASCII file map first to orient yourself to the available paths.
    Do not wrap code in backticks; only raw Python.
    Do not import pandas or numpy; use built-ins only.

    Files like "photo.jpg.md" or "audio.mp3.md" are LLM descriptions/transcriptions of images/audio that were originally in the directory but have been converted to md to make them searchable by you. 
    """

    directory: dict = dspy.InputField(
        desc=(
            "Mapping from relative path to full file text. Keys are the only valid paths. "
            "Use this as the ground-truth corpus."
        )
    )
    file_map: str = dspy.InputField(
        desc=(
            "ASCII tree of directory keys for orientation. Read this first."
        )
    )
    query: str = dspy.InputField(
        desc="User query to find relevant lines. Interpret freely."
    )

    matches: list[Match] = dspy.OutputField(
        desc=(
            "Return match objects with keys: path, line. "
            "path must exactly match a key in directory. line is 1-based. "
            "Return all matches; do not truncate."
        )
    )


class RLMGrepAnswerSignature(dspy.Signature):
    """
    You are the search engine for rlmgrep, a grep-shaped CLI for coding agents.
    Inputs include a directory mapping of files (path-to-text), an ASCII file
    map, and a user query string. Your output must be grep-printable matches as
    (path, line) pairs that point to real lines in the provided texts.
    The query may be natural language or a short pattern; interpret it freely to
    find relevant lines. Return all relevant matches you can find, avoid duplicates,
    and only use exact paths from the directory keys.
    Always read the ASCII file map first to orient yourself to the available paths.
    Do not wrap code in backticks; only raw Python.
    Do not import pandas or numpy; use built-ins only.

    In this mode you are also responsible for generating a narrative answer to the query based on the provided files.

    Files like "photo.jpg.md" or "audio.mp3.md" are LLM descriptions/transcriptions of images/audio that were originally in the directory but have been converted to md to make them searchable by you.

    """

    directory: dict = dspy.InputField(
        desc=(
            "Mapping from relative path to full file text. Keys are the only valid paths. "
            "Use this as the ground-truth corpus."
        )
    )
    file_map: str = dspy.InputField(
        desc=("ASCII tree of directory keys for orientation. Read this first.")
    )
    query: str = dspy.InputField(
        desc="User query to find relevant lines. Interpret freely."
    )

    answer: str = dspy.OutputField(
        desc="Narrative answer to the query based on the provided files."
    )
    matches: list[Match] = dspy.OutputField(
        desc=(
            "Return match objects with keys: path, line. "
            "path must exactly match a key in directory. line is 1-based. "
            "Return all matches; do not truncate."
        )
    )


_CUSTOM_SIGNATURE_PREFIX = "directory: dict, file_map: str, query: str -> "
_CUSTOM_SIGNATURE_INSTRUCTIONS = """
You are the search engine for rlmgrep, a grep-shaped CLI for coding agents.
Inputs include a directory mapping of files (path-to-text), an ASCII file
map, and a user query string.
The query may be natural language or a short pattern; interpret it freely to
extract the requested output fields from the provided files.
Always read the ASCII file map first to orient yourself to the available paths.
Only use exact paths from the directory keys when paths are requested.
Do not wrap code in backticks; only raw Python.
Do not import pandas or numpy; use built-ins only.

Files like "photo.jpg.md" or "audio.mp3.md" are LLM descriptions/transcriptions of images/audio that were originally in the directory but have been converted to md to make them searchable by you.

Return all declared output fields and ensure every value is JSON-compatible.
"""


def _is_supported_signature_type(annotation: Any) -> bool:
    if annotation in {str, int, float, bool, list, dict}:
        return True

    origin = get_origin(annotation)
    if origin is None:
        return False

    if origin is Literal:
        literal_values = get_args(annotation)
        if not literal_values:
            return False
        return all(isinstance(v, (str, int, float, bool)) for v in literal_values)

    if origin is list:
        args = get_args(annotation)
        if not args:
            return True
        if len(args) != 1:
            return False
        return _is_supported_signature_type(args[0])

    if origin is dict:
        args = get_args(annotation)
        if not args:
            return True
        if len(args) != 2:
            return False
        key_type, value_type = args
        if key_type is not str:
            return False
        return _is_supported_signature_type(value_type)

    return False


def build_custom_signature(output_signature: str) -> type[dspy.Signature]:
    if "->" in output_signature:
        raise ValueError("--signature accepts output fields only")

    rhs = output_signature.strip()
    if not rhs:
        raise ValueError("--signature cannot be empty")

    full_signature = f"{_CUSTOM_SIGNATURE_PREFIX}{rhs}"
    try:
        signature = dspy.Signature(full_signature)
    except Exception as exc:  # pragma: no cover - depends on dspy parser internals.
        raise ValueError(f"invalid --signature: {exc}") from exc

    for name, field in signature.output_fields.items():
        annotation = field.annotation
        if not _is_supported_signature_type(annotation):
            raise ValueError(
                f"unsupported output type for field '{name}': {annotation}. "
                "Supported types: str, int, float, bool, list[T], dict[str, T], Literal[...]"
            )

    return signature.with_instructions(_CUSTOM_SIGNATURE_INSTRUCTIONS)


def build_lm(
    model: str | None,
    api_base: str | None,
    api_key: str | None,
    model_type: str | None,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
) -> dspy.LM:
    if not model:
        raise RuntimeError(
            "Model not specified. Set --model or configure ~/.rlmgrep."
        )

    kwargs: dict[str, Any] = {}
    if api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key
    if model_type:
        kwargs["model_type"] = model_type
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    return dspy.LM(model, **kwargs)


def run_rlm(
    directory: dict,
    query: str,
    file_map: str,
    max_iterations: int | None,
    max_llm_calls: int | None,
    verbose: bool,
    sub_lm: dspy.LM | None = None,
    with_answer: bool = False,
) -> tuple[list[Match], str | None]:
    kwargs: dict[str, Any] = {"verbose": verbose}
    if max_iterations is not None:
        kwargs["max_iterations"] = max_iterations
    if max_llm_calls is not None:
        kwargs["max_llm_calls"] = max_llm_calls

    workdir = Path.home() / ".rlmgrep_cache" / "deno"
    workdir.mkdir(parents=True, exist_ok=True)
    interpreter = RLMGrepInterpreter(workdir=workdir)

    signature = RLMGrepAnswerSignature if with_answer else RLMGrepSignature
    rlm = dspy.RLM(signature, sub_lm=sub_lm, interpreter=interpreter, **kwargs)

    try:
        result = rlm(
            directory=directory, file_map=file_map, query=query
        )
    finally:
        interpreter.shutdown()

    matches = list(getattr(result, "matches", []))
    answer = getattr(result, "answer", None) if with_answer else None
    return matches, answer


def run_rlm_with_signature(
    directory: dict,
    query: str,
    file_map: str,
    output_signature: str,
    max_iterations: int | None,
    max_llm_calls: int | None,
    verbose: bool,
    sub_lm: dspy.LM | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"verbose": verbose}
    if max_iterations is not None:
        kwargs["max_iterations"] = max_iterations
    if max_llm_calls is not None:
        kwargs["max_llm_calls"] = max_llm_calls

    workdir = Path.home() / ".rlmgrep_cache" / "deno"
    workdir.mkdir(parents=True, exist_ok=True)
    interpreter = RLMGrepInterpreter(workdir=workdir)

    signature = build_custom_signature(output_signature)
    rlm = dspy.RLM(signature, sub_lm=sub_lm, interpreter=interpreter, **kwargs)

    try:
        result = rlm(directory=directory, file_map=file_map, query=query)
    finally:
        interpreter.shutdown()

    output: dict[str, Any] = {}
    for name in signature.output_fields:
        output[name] = getattr(result, name, None)
    return output
