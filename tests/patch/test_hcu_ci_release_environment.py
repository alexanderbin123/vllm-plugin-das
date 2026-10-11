# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Portable contracts for release-specific HCU CI environment selection."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[2]
CI_SCRIPTS = REPOSITORY / ".github" / "scripts" / "hcu_ci"
if str(CI_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(CI_SCRIPTS))

from resolve_hcu_ci_environment import (  # noqa: E402
    ReleaseEnvironmentError,
    main,
    resolve_release_environment,
)


def test_v0281_ref_resolves_versioned_image_and_checked_out_lock() -> None:
    image = "harbor.example/vllm@sha256:" + "a" * 64

    resolved = resolve_release_environment(
        "v0.28.1-dev",
        {"HCU_CI_IMAGE_V0281": image},
    )

    assert resolved.image == image
    assert resolved.environment_lock == (
        ".github/workflows/configs/hcu-runner-environment.json"
    )


def test_v0281_ref_rejects_missing_versioned_image() -> None:
    with pytest.raises(
        ReleaseEnvironmentError,
        match="HCU_CI_IMAGE_V0281 is required for v0.28.1-dev",
    ):
        resolve_release_environment("v0.28.1-dev", {})


def test_resolver_rejects_unsupported_release_ref() -> None:
    with pytest.raises(
        ReleaseEnvironmentError,
        match="unsupported HCU CI release ref: v0.25.1",
    ):
        resolve_release_environment(
            "v0.25.1",
            {"HCU_CI_IMAGE_V0281": "harbor.example/vllm:v0.28.1"},
        )


def test_cli_failure_does_not_write_partial_github_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "github-output"
    monkeypatch.delenv("HCU_CI_IMAGE_V0281", raising=False)

    assert main(["--ref", "v0.28.1-dev", "--github-output", str(output)]) == 2
    assert not output.exists()
    assert "HCU_CI_IMAGE_V0281 is required" in capsys.readouterr().err


def test_cli_writes_complete_github_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "github-output"
    monkeypatch.setenv("HCU_CI_IMAGE_V0281", "harbor.example/vllm:v0.28.1")

    assert main(["--ref", "v0.28.1-dev", "--github-output", str(output)]) == 0
    assert output.read_text(encoding="utf-8").splitlines() == [
        "image=harbor.example/vllm:v0.28.1",
        "environment_lock=.github/workflows/configs/hcu-runner-environment.json",
    ]
