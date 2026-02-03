from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Callable

from dspy.primitives.code_interpreter import CodeInterpreterError
from dspy.primitives.python_interpreter import PythonInterpreter


class RLMGrepInterpreter(PythonInterpreter):
    def __init__(
        self,
        workdir: Path,
        tools: dict[str, Callable[..., str]] | None = None,
        output_fields: list[dict] | None = None,
        allow_net: list[str] | None = None,
    ) -> None:
        self.workdir = workdir
        self.allow_net = allow_net or ["deno.land", "registry.npmjs.org"]
        deno_command = self._build_deno_command()
        super().__init__(
            deno_command=deno_command,
            tools=tools,
            output_fields=output_fields,
        )

    def _build_deno_command(self) -> list[str]:
        runner_path = self._get_runner_path()
        deno_dir = self._get_deno_dir()
        home_node_modules = Path.home() / "node_modules"
        allowed_read_paths = [runner_path, str(self.workdir), str(home_node_modules)]
        if deno_dir:
            allowed_read_paths.append(deno_dir)

        allow_read = f"--allow-read={','.join(allowed_read_paths)}"
        write_paths = [str(self.workdir), str(home_node_modules)]
        allow_write = f"--allow-write={','.join(write_paths)}"
        allow_net = f"--allow-net={','.join(self.allow_net)}"

        return [
            "deno",
            "run",
            "--node-modules-dir=auto",
            allow_read,
            allow_write,
            allow_net,
            runner_path,
        ]

    def _ensure_deno_process(self) -> None:
        if self.deno_process is None or self.deno_process.poll() is not None:
            try:
                env = os.environ.copy()
                env.setdefault("DENO_DIR", str(self.workdir))
                self.deno_process = subprocess.Popen(
                    self.deno_command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="UTF-8",
                    env=env,
                    cwd=str(self.workdir),
                )
            except FileNotFoundError as exc:
                raise CodeInterpreterError(
                    "Deno not found. Install Deno: https://docs.deno.com/runtime/getting_started/installation/"
                ) from exc
