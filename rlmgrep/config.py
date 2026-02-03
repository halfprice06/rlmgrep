from __future__ import annotations

from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib as _tomllib  # type: ignore
except Exception:  # pragma: no cover - fallback
    import tomli as _tomllib  # type: ignore


DEFAULT_CONFIG_TEXT = "\n".join(
    [
        'model = "openai/gpt-5.2"',
        'sub_model = "openai/gpt-5-mini"',
        'api_base = "https://api.openai.com/v1"',
        'model_type = "responses"',
        "temperature = 1.0",
        "max_tokens = 64000",
        "max_iterations = 10",
        "max_llm_calls = 20",
        "# markitdown_enable_images = false",
        "# markitdown_image_llm_model = \"gpt-5-mini\"",
        "# markitdown_image_llm_provider = \"openai\"",
        "# markitdown_image_llm_api_key = \"\"",
        "# markitdown_image_llm_api_base = \"\"",
        "# markitdown_image_llm_prompt = \"\"",
        "# markitdown_enable_audio = false",
        "# markitdown_audio_model = \"gpt-4o-mini-transcribe\"",
        "# markitdown_audio_provider = \"openai\"",
        "# markitdown_audio_api_key = \"\"",
        "# markitdown_audio_api_base = \"\"",
        "",
    ]
)


def ensure_default_config(path: Path | None = None) -> tuple[Path | None, str | None]:
    config_path = path or (Path.home() / ".rlmgrep")
    if config_path.exists():
        return None, None
    try:
        config_path.write_text(DEFAULT_CONFIG_TEXT)
    except Exception as exc:  # pragma: no cover - defensive
        return None, str(exc)
    return config_path, None


def load_config(path: Path | None = None) -> tuple[dict[str, Any], list[str]]:
    config_path = path or (Path.home() / ".rlmgrep")
    if not config_path.exists():
        return {}, []
    if not config_path.is_file():
        return {}, [f"config path is not a file: {config_path}"]

    try:
        data = _tomllib.loads(config_path.read_text())
    except Exception as exc:  # pragma: no cover - defensive
        return {}, [f"failed to read config {config_path}: {exc}"]

    if not isinstance(data, dict):
        return {}, [f"config file {config_path} is not a table"]
    return data, []
