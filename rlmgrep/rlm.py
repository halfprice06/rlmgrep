from __future__ import annotations

from pathlib import Path
from typing import Any

import dspy
from pydantic import BaseModel

from .interpreter import RLMGrepInterpreter


class Match(BaseModel):
    path: str
    line: int


class RLMGrepSignature(dspy.Signature):
    """
    You are the search engine for rlmgrep, a grep-shaped CLI for coding agents.
    Inputs include a directory mapping of files (path -> full text), an ASCII file
    map, and a user query string. Your output must be grep-printable matches as
    (path, line) pairs that point to real lines in the provided texts.
    The query may be natural language or a short pattern; interpret it freely to
    find relevant lines. Return all relevant matches you can find, avoid duplicates,
    and only use exact paths from the directory keys.
    Always read the ASCII file map first to orient yourself to the available paths.
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
) -> list[Match]:
    kwargs: dict[str, Any] = {"verbose": verbose}
    if max_iterations is not None:
        kwargs["max_iterations"] = max_iterations
    if max_llm_calls is not None:
        kwargs["max_llm_calls"] = max_llm_calls

    workdir = Path.home() / ".rlmgrep_cache" / "deno"
    workdir.mkdir(parents=True, exist_ok=True)
    interpreter = RLMGrepInterpreter(workdir=workdir)

    rlm = dspy.RLM(RLMGrepSignature, sub_lm=sub_lm, interpreter=interpreter, **kwargs)

    try:
        result = rlm(
            directory=directory, file_map=file_map, query=query
        )
    finally:
        interpreter.shutdown()

    return list(getattr(result, "matches", []))
