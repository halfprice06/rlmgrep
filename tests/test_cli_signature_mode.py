from __future__ import annotations

import io
import json
import sys

import pytest

import rlmgrep.cli as cli


class _FakeStdin(io.StringIO):
    def isatty(self) -> bool:
        return False


def _patch_signature_mode_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "ensure_default_config", lambda: (None, None))
    monkeypatch.setattr(cli, "load_config", lambda: ({}, []))
    monkeypatch.setattr(
        cli,
        "_build_markitdown",
        lambda config, warnings: (None, False, False, None),
    )
    monkeypatch.setattr(cli, "build_lm", lambda **kwargs: object())
    monkeypatch.setattr(cli, "_pick_api_key", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli.dspy, "configure", lambda **kwargs: None)
    monkeypatch.setattr(sys, "stdin", _FakeStdin("stdin content for signature mode"))


def test_signature_json_outputs_compact_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _patch_signature_mode_dependencies(monkeypatch)
    captured_signature: dict[str, str] = {}

    def _fake_run_rlm_with_signature(**kwargs):
        captured_signature["value"] = kwargs["output_signature"]
        return {
            "summary": "done",
            "findings": [{"path": "src/auth.py", "line": 42}],
        }

    def _unexpected_run_rlm(**kwargs):
        raise AssertionError("run_rlm should not be called in signature mode")

    monkeypatch.setattr(cli, "run_rlm_with_signature", _fake_run_rlm_with_signature)
    monkeypatch.setattr(cli, "run_rlm", _unexpected_run_rlm)

    exit_code = cli.main(
        [
            "--signature-json",
            "summary: str, findings: list[dict[str, int]]",
            "audit auth",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert (
        captured.out.strip()
        == '{"summary":"done","findings":[{"path":"src/auth.py","line":42}]}'
    )
    assert (
        captured_signature["value"]
        == "summary: str, findings: list[dict[str, int]]"
    )


def test_signature_markdown_is_default_for_signature_flag(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_signature_mode_dependencies(monkeypatch)

    monkeypatch.setattr(
        cli,
        "run_rlm_with_signature",
        lambda **kwargs: {
            "summary": "Auth checks are inconsistent",
            "ok": True,
            "findings": [{"path": "src/auth.py", "line": 42, "severity": "high"}],
        },
    )

    exit_code = cli.main(
        [
            "--signature",
            "summary: str, ok: bool, findings: list[dict[str, str]]",
            "audit auth",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "===== Signature Output =====" in captured.out
    assert "----- summary -----\nAuth checks are inconsistent" in captured.out
    assert "----- ok -----\ntrue" in captured.out
    assert "----- findings -----" in captured.out


def test_signature_flags_conflict_with_answer_mode(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["--signature", "summary: str", "--answer", "query"])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert (
        "--signature/--signature-json cannot be combined with --answer or --answer-only"
        in captured.err
    )


def test_signature_mode_surfaces_signature_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_signature_mode_dependencies(monkeypatch)
    monkeypatch.setattr(
        cli,
        "run_rlm_with_signature",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("bad signature")),
    )

    exit_code = cli.main(["--signature", "summary: str", "query"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "bad signature" in captured.err


def test_signature_json_serializes_non_json_python_values(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_signature_mode_dependencies(monkeypatch)

    class _CustomObject:
        def __str__(self) -> str:
            return "custom-object"

    monkeypatch.setattr(
        cli,
        "run_rlm_with_signature",
        lambda **kwargs: {"tags": {"a", "b"}, "meta": _CustomObject()},
    )

    exit_code = cli.main(["--signature-json", "tags: list[str], meta: str", "query"])

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert exit_code == 0
    assert set(data["tags"]) == {"a", "b"}
    assert data["meta"] == "custom-object"


def test_signature_mode_returns_zero_even_for_empty_payload(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_signature_mode_dependencies(monkeypatch)
    monkeypatch.setattr(cli, "run_rlm_with_signature", lambda **kwargs: {})

    exit_code = cli.main(["--signature", "summary: str", "query"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "No output fields returned." in captured.out
