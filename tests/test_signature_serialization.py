from __future__ import annotations

from pydantic import BaseModel

from rlmgrep.cli import _render_signature_markdown, _to_json_compatible


class Finding(BaseModel):
    path: str
    line: int


class Report(BaseModel):
    summary: str
    findings: list[Finding]


class _CustomObject:
    def __str__(self) -> str:
        return "custom-object"


def test_to_json_compatible_converts_nested_values() -> None:
    payload = {
        7: Report(
            summary="ok",
            findings=[Finding(path="a.py", line=3), Finding(path="b.py", line=9)],
        ),
        "mixed": ({"x", "y"}, _CustomObject()),
        "none": None,
    }

    converted = _to_json_compatible(payload)

    assert converted["7"]["summary"] == "ok"
    assert converted["7"]["findings"] == [
        {"path": "a.py", "line": 3},
        {"path": "b.py", "line": 9},
    ]
    assert sorted(converted["mixed"][0]) == ["x", "y"]
    assert converted["mixed"][1] == "custom-object"
    assert converted["none"] is None


def test_render_signature_markdown_formats_scalars_and_structures() -> None:
    payload = {
        "summary": "Authentication checks are inconsistent",
        "ok": True,
        "count": 2,
        "missing": None,
        "findings": [{"path": "src/auth.py", "line": 42, "severity": "high"}],
    }

    rendered = _render_signature_markdown(payload)

    assert rendered.startswith("===== Signature Output =====")
    assert "----- summary -----\nAuthentication checks are inconsistent" in rendered
    assert "----- ok -----\ntrue" in rendered
    assert "----- count -----\n2" in rendered
    assert "----- missing -----\nnull" in rendered
    assert "----- findings -----" in rendered
    assert '"path": "src/auth.py"' in rendered


def test_render_signature_markdown_handles_empty_payload() -> None:
    rendered = _render_signature_markdown({})
    assert "No output fields returned." in rendered
