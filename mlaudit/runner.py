"""Run the audited platform's own commands inside disposable copies."""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

COPY_IGNORE = shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", ".ruff_cache")


@dataclass
class RunResult:
    name: str
    command: str
    returncode: int
    seconds: float
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def brief(self) -> dict:
        return {"name": self.name, "command": self.command, "returncode": self.returncode,
                "seconds": round(self.seconds, 1),
                "stderr_tail": self.stderr.strip().splitlines()[-6:]}


def make_sandbox(src: Path, dest: Path) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest, ignore=COPY_IGNORE)
    return dest


def run(name: str, template: str, cwd: Path, python: str, env_extra: dict | None = None,
        timeout: int = 900, **fmt) -> RunResult:
    cmd = template.format(python=python, **fmt)
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"})
    env.pop("PYTHONPATH", None)
    if env_extra:
        env.update(env_extra)
    t0 = time.time()
    try:
        p = subprocess.run(shlex.split(cmd), cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
        rc, so, se = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as exc:
        rc, so, se = 124, exc.stdout or "", (exc.stderr or "") + f"\nTIMEOUT after {timeout} s"
        so = so.decode() if isinstance(so, bytes) else so
        se = se.decode() if isinstance(se, bytes) else se
    return RunResult(name, cmd, rc, time.time() - t0, so, se)
