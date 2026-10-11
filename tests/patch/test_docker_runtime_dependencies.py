from __future__ import annotations

import os
import subprocess
from pathlib import Path

from packaging.requirements import Requirement


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
INSTALLER = (
    REPOSITORY_ROOT / "scripts/ci/hcu/install_vllm_runtime_dependencies.sh"
)
REQUIREMENTS = REPOSITORY_ROOT / "docker/requirements-vllm-runtime.txt"


def _recording_python(tmp_path: Path) -> tuple[Path, Path]:
    executable = tmp_path / "record-python"
    argv_log = tmp_path / "argv"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\0' \"$@\" > \"$RECORDED_ARGV\"\n"
        "exit \"${FAKE_PIP_EXIT:-0}\"\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable, argv_log


def _run_installer(
    tmp_path: Path,
    *arguments: str,
    pip_exit: int = 0,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    python, argv_log = _recording_python(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "PYTHON": str(python),
            "RECORDED_ARGV": str(argv_log),
            "FAKE_PIP_EXIT": str(pip_exit),
        }
    )
    result = subprocess.run(
        ["bash", str(INSTALLER), *arguments],
        cwd=REPOSITORY_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    argv = []
    if argv_log.exists():
        argv = argv_log.read_bytes().rstrip(b"\0").decode().split("\0")
    return result, argv


def test_installer_uses_active_pip_configuration(tmp_path: Path) -> None:
    supplied_requirements = tmp_path / "runtime requirements.txt"
    supplied_requirements.touch()

    result, argv = _run_installer(tmp_path, str(supplied_requirements))

    assert result.returncode == 0, result.stderr
    assert argv == [
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--requirement",
        str(supplied_requirements),
    ]
    assert "--index-url" not in argv
    assert "--extra-index-url" not in argv


def test_installer_defaults_to_repository_runtime_requirements(
    tmp_path: Path,
) -> None:
    result, argv = _run_installer(tmp_path)

    assert result.returncode == 0, result.stderr
    assert argv[-2:] == ["--requirement", str(REQUIREMENTS)]


def test_installer_propagates_pip_failure(tmp_path: Path) -> None:
    supplied_requirements = tmp_path / "runtime.txt"
    supplied_requirements.touch()

    result, _ = _run_installer(
        tmp_path,
        str(supplied_requirements),
        pip_exit=37,
    )

    assert result.returncode == 37


def test_runtime_requirements_pin_ray_extras_and_pandas() -> None:
    requirements = {
        requirement.name: requirement
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if (stripped := line.strip()) and not stripped.startswith("#")
        for requirement in [Requirement(stripped)]
    }

    ray = requirements["ray"]
    assert ray.extras == {"data", "serve", "train", "tune"}
    assert str(ray.specifier) == "==2.58.0"
    assert str(requirements["pandas"].specifier) == "==2.3.3"
