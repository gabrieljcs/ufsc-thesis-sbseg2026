# Forwards thesis-eval arguments into the GPU Docker container. Registered as
# the thesis-eval-gpu console script.
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_IMAGE = "thesis-eval:gpu"


def _find_repo_root() -> Path:
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    sys.exit(
        "thesis-eval-gpu: no pyproject.toml at or above cwd; "
        "run this from inside the thesis-eval repo."
    )


def main() -> None:
    if not shutil.which("docker"):
        sys.exit(
            "thesis-eval-gpu: docker not found on PATH. Install Docker Desktop "
            "(Windows/Mac) or Docker Engine + nvidia-container-toolkit (Linux), "
            "then build the GPU image:\n"
            "  docker build -f docker/Dockerfile.gpu -t thesis-eval:gpu .\n"
            "See the Docker GPU image section in README.md for details."
        )

    image = os.environ.get("THESIS_EVAL_GPU_IMAGE", DEFAULT_IMAGE)
    repo = _find_repo_root()

    cmd: list[str] = ["docker", "run", "--rm", "--gpus", "all"]
    if sys.stdin.isatty() and sys.stdout.isatty():
        cmd.append("-it")
    cmd.extend(["-v", f"{repo}:/work/eval", "-w", "/work/eval"])

    env_file = repo / ".env"
    if env_file.exists():
        cmd.extend(["--env-file", str(env_file)])

    cmd.append(image)
    cmd.extend(["uv", "run", "thesis-eval", *sys.argv[1:]])

    try:
        sys.exit(subprocess.call(cmd))
    except KeyboardInterrupt:
        sys.exit(130)
