from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import Iterable, Any, Callable

from pypdf import PdfReader


@dataclass
class FileRecord:
    path: str
    text: str
    lines: list[str]
    page_map: list[int] | None = None


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _is_binary(data: bytes) -> bool:
    # Heuristic: presence of NUL byte.
    return b"\x00" in data


def _read_pdf(path: Path) -> tuple[str, list[int]]:
    reader = PdfReader(str(path))
    lines: list[str] = []
    page_map: list[int] = []
    for idx, page in enumerate(reader.pages, start=1):
        lines.append(f"===== Page {idx} =====")
        page_map.append(idx)
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = _normalize_newlines(text)
        if text:
            for line in text.split("\n"):
                lines.append(line)
                page_map.append(idx)
        else:
            lines.append("")
            page_map.append(idx)
    return "\n".join(lines), page_map


IMAGE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".heic",
    ".heif",
}

AUDIO_EXTS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".opus",
}


def _sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".md")


def _read_sidecar(path: Path) -> str | None:
    sidecar = _sidecar_path(path)
    if sidecar.is_file():
        return _normalize_newlines(sidecar.read_text(errors="replace"))
    return None


def _write_sidecar(path: Path, text: str) -> None:
    try:
        _sidecar_path(path).write_text(text)
    except Exception:
        pass


def _load_file(
    path: Path,
    markitdown: Any | None = None,
    enable_images: bool = False,
    enable_audio: bool = False,
    audio_transcriber: Callable[[Path], str] | None = None,
    binary_as_text: bool = False,
) -> tuple[str | None, list[int] | None, str | None]:
    """
    Returns (text, page_map, error). If error is not None, text will be None.
    """
    suffix = path.suffix.lower()
    is_image = suffix in IMAGE_EXTS
    is_audio = suffix in AUDIO_EXTS
    try:
        if suffix == ".pdf":
            text, page_map = _read_pdf(path)
            return text, page_map, None

        data = path.read_bytes()
        if not _is_binary(data) or binary_as_text:
            return _normalize_newlines(data.decode("utf-8", errors="replace")), None, None

        if (is_image and enable_images) or (is_audio and enable_audio):
            sidecar_text = _read_sidecar(path)
            if sidecar_text is not None:
                return sidecar_text, None, None

        if markitdown is None:
            return None, None, "binary file"

        if is_image and not enable_images:
            return None, None, "image conversion disabled"
        if is_audio and not enable_audio:
            return None, None, "audio conversion disabled"

        if is_audio and audio_transcriber is not None:
            try:
                text = _normalize_newlines(audio_transcriber(path))
            except Exception as exc:
                return None, None, f"audio transcription failed: {exc}"
            _write_sidecar(path, text)
            return text, None, None

        result = markitdown.convert(str(path))
        text = _normalize_newlines(getattr(result, "text_content", "") or "")

        if is_image or is_audio:
            _write_sidecar(path, text)

        return text, None, None
    except Exception as exc:  # pragma: no cover - defensive
        return None, None, str(exc)


def collect_files(paths: Iterable[str], recursive: bool = True) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if not p.exists():
            continue
        if p.is_dir():
            if recursive:
                files.extend(fp for fp in p.rglob("*") if fp.is_file())
            else:
                # No recursion: ignore directories.
                continue
        elif p.is_file():
            files.append(p)
    return files


TYPE_EXTS = {
    "bash": {".bash"},
    "c": {".c", ".h"},
    "cpp": {".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx"},
    "conf": {".conf"},
    "css": {".css"},
    "csv": {".csv"},
    "cs": {".cs"},
    "env": {".env"},
    "eslintrc": {".eslintrc"},
    "fish": {".fish"},
    "gitignore": {".gitignore"},
    "go": {".go"},
    "html": {".html", ".htm"},
    "ini": {".ini"},
    "java": {".java"},
    "js": {".js", ".mjs", ".cjs"},
    "json": {".json"},
    "kt": {".kt", ".kts"},
    "less": {".less"},
    "log": {".log"},
    "md": {".md"},
    "php": {".php"},
    "py": {".py", ".pyi"},
    "rb": {".rb"},
    "rs": {".rs"},
    "rst": {".rst"},
    "sass": {".sass"},
    "scala": {".scala"},
    "scss": {".scss"},
    "sh": {".sh"},
    "sql": {".sql"},
    "svg": {".svg"},
    "swift": {".swift"},
    "tex": {".tex"},
    "toml": {".toml"},
    "ts": {".ts", ".tsx", ".mts", ".cts"},
    "tsv": {".tsv"},
    "txt": {".txt"},
    "xml": {".xml"},
    "yaml": {".yaml", ".yml"},
    "zsh": {".zsh"},
}


def resolve_type_exts(type_names: Iterable[str]) -> tuple[set[str], list[str]]:
    exts: set[str] = set()
    warnings: list[str] = []
    for raw in type_names:
        name = raw.strip().lower()
        if not name:
            continue
        if name in TYPE_EXTS:
            exts.update(TYPE_EXTS[name])
            continue
        if name.startswith("."):
            exts.add(name)
            continue
        # Unknown type: treat as extension but warn.
        exts.add("." + name)
        warnings.append(f"unknown type '{raw}', treating as extension '.{name}'")
    return exts, warnings


def _matches_globs(path: str, globs: list[str]) -> bool:
    if not globs:
        return True
    posix = path.replace("\\", "/")
    p = PurePosixPath(posix)
    for pattern in globs:
        pat = pattern.replace("\\", "/")
        if p.match(pat) or fnmatch(posix, pat):
            return True
    return False


def load_files(
    paths: Iterable[str],
    cwd: Path,
    recursive: bool = True,
    include_globs: list[str] | None = None,
    type_exts: set[str] | None = None,
    markitdown: Any | None = None,
    enable_images: bool = False,
    enable_audio: bool = False,
    audio_transcriber: Callable[[Path], str] | None = None,
    binary_as_text: bool = False,
) -> tuple[dict[str, FileRecord], list[str]]:
    records: dict[str, FileRecord] = {}
    warnings: list[str] = []
    image_convert_count = 0
    audio_convert_count = 0

    files = collect_files(paths, recursive=recursive)
    for fp in files:
        try:
            key = fp.relative_to(cwd).as_posix()
        except ValueError:
            key = fp.as_posix()

        if include_globs and not _matches_globs(key, include_globs):
            continue

        if type_exts:
            if fp.suffix.lower() not in type_exts:
                continue

        suffix = fp.suffix.lower()
        if markitdown is not None and not binary_as_text:
            if enable_images and suffix in IMAGE_EXTS:
                if not _sidecar_path(fp).is_file():
                    image_convert_count += 1
            if enable_audio and suffix in AUDIO_EXTS:
                if not _sidecar_path(fp).is_file():
                    audio_convert_count += 1

        text, page_map, err = _load_file(
            fp,
            markitdown=markitdown,
            enable_images=enable_images,
            enable_audio=enable_audio,
            audio_transcriber=audio_transcriber,
            binary_as_text=binary_as_text,
        )
        if err is not None:
            silent_errors = {
                "binary file",
                "image conversion disabled",
                "audio conversion disabled",
            }
            if err not in silent_errors and "No converter attempted a conversion" not in err:
                warnings.append(f"skip {fp}: {err}")
            continue
        if text is None:
            warnings.append(f"skip {fp}: unreadable")
            continue

        lines = text.split("\n")
        if page_map is not None and len(page_map) != len(lines):
            page_map = None
        records[key] = FileRecord(path=key, text=text, lines=lines, page_map=page_map)

    if image_convert_count > 5:
        warnings.append(
            f"markitdown converting {image_convert_count} images (over 5)"
        )
    if audio_convert_count > 5:
        warnings.append(
            f"markitdown converting {audio_convert_count} audio files (over 5)"
        )

    return records, warnings
