from __future__ import annotations

import pytest

from rlmgrep.rlm import build_custom_signature


@pytest.mark.parametrize(
    ("output_signature", "expected_fields"),
    [
        ("summary", ["summary"]),
        ("summary: str", ["summary"]),
        ("count: int, confidence: float, ok: bool", ["count", "confidence", "ok"]),
        ("files: list[str]", ["files"]),
        ("scores: list[int]", ["scores"]),
        ("metrics: dict[str, float]", ["metrics"]),
        ("records: list[dict[str, str]]", ["records"]),
        ("nested: dict[str, list[dict[str, int]]]", ["nested"]),
        (
            'severities: list[Literal["low","medium","high"]], summary: str',
            ["severities", "summary"],
        ),
        ('choice: Literal["ok", 1, 2.5, False]', ["choice"]),
        ("bare_list: list, bare_dict: dict", ["bare_list", "bare_dict"]),
        (
            'report: dict[str, list[dict[str, Literal["pass","fail"]]]]',
            ["report"],
        ),
    ],
)
def test_build_custom_signature_accepts_supported_types(
    output_signature: str,
    expected_fields: list[str],
) -> None:
    signature = build_custom_signature(output_signature)

    assert list(signature.input_fields.keys()) == ["directory", "file_map", "query"]
    assert list(signature.output_fields.keys()) == expected_fields
    assert "JSON-compatible" in signature.instructions


@pytest.mark.parametrize(
    ("output_signature", "expected_error"),
    [
        ("", "--signature cannot be empty"),
        ("   ", "--signature cannot be empty"),
        ("x: str -> y: str", "output fields only"),
        ("items: tuple[str, int]", "unsupported output type"),
        ("items: set[str]", "unsupported output type"),
        ("data: dict[int, str]", "unsupported output type"),
        ("value: Union[int, str]", "unsupported output type"),
        ("values: list[Union[int, str]]", "unsupported output type"),
        ("matrix: list[tuple[int, int]]", "unsupported output type"),
        ("choice: Literal[None]", "unsupported output type"),
        ("value: complex", "unsupported output type"),
        ("a::int", "invalid --signature"),
    ],
)
def test_build_custom_signature_rejects_unsupported_or_invalid_types(
    output_signature: str,
    expected_error: str,
) -> None:
    with pytest.raises(ValueError, match=expected_error):
        build_custom_signature(output_signature)

