from __future__ import annotations

from pathlib import Path

from rlmgrep.cli import _needs_markitdown


def test_needs_markitdown_false_for_text_only_files(tmp_path: Path) -> None:
    text_file = tmp_path / "notes.md"
    text_file.write_text("hello\nworld\n", encoding="utf-8")

    assert _needs_markitdown([text_file], binary_as_text=False) is False


def test_needs_markitdown_true_for_known_binary_extensions(tmp_path: Path) -> None:
    docx_file = tmp_path / "report.docx"
    docx_file.write_text("placeholder", encoding="utf-8")

    assert _needs_markitdown([docx_file], binary_as_text=False) is True


def test_needs_markitdown_true_for_binary_blob(tmp_path: Path) -> None:
    blob = tmp_path / "blob.bin"
    blob.write_bytes(b"\x00\x01\x02")

    assert _needs_markitdown([blob], binary_as_text=False) is True


def test_needs_markitdown_false_for_binary_when_binary_as_text(tmp_path: Path) -> None:
    blob = tmp_path / "blob.bin"
    blob.write_bytes(b"\x00\x01\x02")

    assert _needs_markitdown([blob], binary_as_text=True) is False


def test_needs_markitdown_false_for_pdf(tmp_path: Path) -> None:
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.7")

    assert _needs_markitdown([pdf], binary_as_text=False) is False

