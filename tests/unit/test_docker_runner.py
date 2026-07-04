"""Unit tests for bernstein.core.docker_runner.DockerRunner.

All Docker/git interactions are mocked via unittest.mock.patch on
subprocess.run (or on the git_ops.worktree_add helper) — no real `docker`
or `git` commands are ever invoked, per this suite's hermetic-unit-test
convention (see tests/unit/conftest.py / _no_network.py).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from bernstein.core.docker_runner import (
    ContainerInfo,
    DockerRunner,
    DockerRunnerError,
    RunStatus,
)
from bernstein.core.git.git_ops import GitResult


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["docker"], returncode=returncode, stdout=stdout, stderr=stderr)


@pytest.fixture
def repo_path(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    return root


@pytest.fixture
def runner(repo_path: Path) -> DockerRunner:
    return DockerRunner(repo_path=repo_path, port_range=(18000, 18002))


# ---------------------------------------------------------------------------
# find_available_port
# ---------------------------------------------------------------------------


def test_find_available_port_returns_first_free(runner: DockerRunner) -> None:
    # 18000 is bound, 18001 and 18002 are free -> expect 18001.
    ps_output = "0.0.0.0:18000->18000/tcp, :::18000->18000/tcp\n"
    with patch("bernstein.core.docker_runner._run_subprocess", return_value=_completed(0, stdout=ps_output)):
        port = runner.find_available_port()
    assert port == 18001


def test_find_available_port_all_free_returns_range_start(runner: DockerRunner) -> None:
    with patch("bernstein.core.docker_runner._run_subprocess", return_value=_completed(0, stdout="")):
        port = runner.find_available_port()
    assert port == 18000


def test_find_available_port_range_exhausted_raises(runner: DockerRunner) -> None:
    ps_output = (
        "0.0.0.0:18000->18000/tcp\n"
        "0.0.0.0:18001->18001/tcp\n"
        "0.0.0.0:18002->18002/tcp\n"
    )
    with patch("bernstein.core.docker_runner._run_subprocess", return_value=_completed(0, stdout=ps_output)):
        with pytest.raises(DockerRunnerError, match="exhausted"):
            runner.find_available_port()


def test_find_available_port_docker_ps_failure_raises(runner: DockerRunner) -> None:
    with patch(
        "bernstein.core.docker_runner._run_subprocess",
        return_value=_completed(1, stderr="docker daemon not running"),
    ):
        with pytest.raises(DockerRunnerError, match="docker daemon not running"):
            runner.find_available_port()


# ---------------------------------------------------------------------------
# Container naming
# ---------------------------------------------------------------------------


def test_container_name_format_matches_list_active_filter(runner: DockerRunner, repo_path: Path, tmp_path: Path) -> None:
    """bernstein-<run-id> is deterministic and matches list_active_runs()'s filter prefix."""
    run_id = "issue-42-fix-auth-20260704-a1b2c3d4"
    workflow_yaml = tmp_path / "bernstein.yaml"
    workflow_yaml.write_text("goal: test\n")
    secrets_env = tmp_path / "secrets.env"
    secrets_env.write_text("ANTHROPIC_API_KEY=x\n")

    with (
        patch("bernstein.core.docker_runner.worktree_add", return_value=GitResult(0, "", "")),
        patch.object(runner, "find_available_port", return_value=18042),
        patch(
            "bernstein.core.docker_runner._run_subprocess",
            return_value=_completed(0, stdout="deadbeef1234\n"),
        ),
    ):
        # worktree_add is mocked so it never actually creates the directory;
        # create it manually so shutil.copy2 has somewhere to land.
        worktree_path = repo_path / ".sdd" / "worktrees" / run_id
        worktree_path.mkdir(parents=True)
        info = runner.launch_run(run_id, workflow_yaml, secrets_env)

    assert info.container_name == f"bernstein-{run_id}"
    assert info.container_name.startswith("bernstein-")
    assert info.run_id == run_id
    assert info.port == 18042
    assert info.container_id == "deadbeef1234"


# ---------------------------------------------------------------------------
# poll_status / inspect_health parsing
# ---------------------------------------------------------------------------


def _inspect_result(state: dict) -> subprocess.CompletedProcess[str]:
    return _completed(0, stdout=json.dumps(state))


def test_poll_status_running(runner: DockerRunner) -> None:
    state = {"Running": True, "ExitCode": 0, "OOMKilled": False, "Error": ""}
    with patch("bernstein.core.docker_runner._run_subprocess", return_value=_inspect_result(state)):
        assert runner.poll_status("abc123") == RunStatus.RUNNING


def test_poll_status_exited_zero(runner: DockerRunner) -> None:
    state = {"Running": False, "ExitCode": 0, "OOMKilled": False, "Error": ""}
    with patch("bernstein.core.docker_runner._run_subprocess", return_value=_inspect_result(state)):
        health = runner.inspect_health("abc123")
    assert health.status == RunStatus.EXITED
    assert health.exit_code == 0
    assert health.oom_killed is False


def test_poll_status_exited_nonzero(runner: DockerRunner) -> None:
    state = {"Running": False, "ExitCode": 137, "OOMKilled": False, "Error": ""}
    with patch("bernstein.core.docker_runner._run_subprocess", return_value=_inspect_result(state)):
        health = runner.inspect_health("abc123")
    assert health.status == RunStatus.EXITED
    assert health.exit_code == 137


def test_poll_status_oom_killed(runner: DockerRunner) -> None:
    state = {"Running": False, "ExitCode": 137, "OOMKilled": True, "Error": ""}
    with patch("bernstein.core.docker_runner._run_subprocess", return_value=_inspect_result(state)):
        health = runner.inspect_health("abc123")
    assert health.status == RunStatus.KILLED
    assert health.oom_killed is True


def test_poll_status_inspect_command_failure_returns_unknown(runner: DockerRunner) -> None:
    with patch(
        "bernstein.core.docker_runner._run_subprocess",
        return_value=_completed(1, stderr="No such container: abc123"),
    ):
        assert runner.poll_status("abc123") == RunStatus.UNKNOWN


def test_poll_status_malformed_json_returns_unknown(runner: DockerRunner) -> None:
    with patch("bernstein.core.docker_runner._run_subprocess", return_value=_completed(0, stdout="not json")):
        assert runner.poll_status("abc123") == RunStatus.UNKNOWN


# ---------------------------------------------------------------------------
# Bad subprocess return code paths
# ---------------------------------------------------------------------------


def test_launch_run_docker_run_nonzero_raises_with_stderr(runner: DockerRunner, repo_path: Path, tmp_path: Path) -> None:
    run_id = "test-run-bad-docker"
    workflow_yaml = tmp_path / "bernstein.yaml"
    workflow_yaml.write_text("goal: test\n")
    secrets_env = tmp_path / "secrets.env"
    secrets_env.write_text("ANTHROPIC_API_KEY=x\n")

    distinctive_stderr = "docker: Error response from daemon: distinctive failure marker"

    with (
        patch("bernstein.core.docker_runner.worktree_add", return_value=GitResult(0, "", "")),
        patch.object(runner, "find_available_port", return_value=18042),
        patch(
            "bernstein.core.docker_runner._run_subprocess",
            return_value=_completed(1, stderr=distinctive_stderr),
        ),
    ):
        worktree_path = repo_path / ".sdd" / "worktrees" / run_id
        worktree_path.mkdir(parents=True)
        with pytest.raises(DockerRunnerError) as exc_info:
            runner.launch_run(run_id, workflow_yaml, secrets_env)

    assert "distinctive failure marker" in str(exc_info.value)


def test_kill_run_nonzero_raises_with_stderr(runner: DockerRunner) -> None:
    distinctive_stderr = "Error: No such container: distinctive-marker-xyz"
    with patch("bernstein.core.docker_runner._run_subprocess", return_value=_completed(1, stderr=distinctive_stderr)):
        with pytest.raises(DockerRunnerError) as exc_info:
            runner.kill_run("distinctive-marker-xyz")
    assert "distinctive-marker-xyz" in str(exc_info.value)


def test_cleanup_nonzero_raises_with_stderr(runner: DockerRunner) -> None:
    distinctive_stderr = "Error: cannot remove container: distinctive cleanup failure"
    with patch("bernstein.core.docker_runner._run_subprocess", return_value=_completed(1, stderr=distinctive_stderr)):
        with pytest.raises(DockerRunnerError) as exc_info:
            runner.cleanup("some-container-id")
    assert "distinctive cleanup failure" in str(exc_info.value)


def test_launch_run_worktree_add_lock_contention_retries_then_succeeds(
    runner: DockerRunner, repo_path: Path, tmp_path: Path
) -> None:
    """git worktree add hits index.lock contention twice, then succeeds."""
    run_id = "test-run-lock-contention"
    workflow_yaml = tmp_path / "bernstein.yaml"
    workflow_yaml.write_text("goal: test\n")
    secrets_env = tmp_path / "secrets.env"
    secrets_env.write_text("ANTHROPIC_API_KEY=x\n")

    worktree_path = repo_path / ".sdd" / "worktrees" / run_id

    call_count = {"n": 0}

    def fake_worktree_add(cwd: Path, path: Path, branch: str) -> GitResult:
        call_count["n"] += 1
        if call_count["n"] < 3:
            return GitResult(128, "", "fatal: Unable to create '.git/index.lock': File exists")
        worktree_path.mkdir(parents=True, exist_ok=True)
        return GitResult(0, "", "")

    with (
        patch("bernstein.core.docker_runner.worktree_add", side_effect=fake_worktree_add),
        patch("bernstein.core.docker_runner.time.sleep", return_value=None),
        patch.object(runner, "find_available_port", return_value=18042),
        patch(
            "bernstein.core.docker_runner._run_subprocess",
            return_value=_completed(0, stdout="deadbeef1234\n"),
        ),
    ):
        info = runner.launch_run(run_id, workflow_yaml, secrets_env)

    assert call_count["n"] == 3
    assert isinstance(info, ContainerInfo)
    assert info.run_id == run_id


def test_launch_run_worktree_add_non_lock_failure_raises_immediately(
    runner: DockerRunner, repo_path: Path, tmp_path: Path
) -> None:
    run_id = "test-run-bad-worktree"
    workflow_yaml = tmp_path / "bernstein.yaml"
    workflow_yaml.write_text("goal: test\n")
    secrets_env = tmp_path / "secrets.env"
    secrets_env.write_text("ANTHROPIC_API_KEY=x\n")

    with patch(
        "bernstein.core.docker_runner.worktree_add",
        return_value=GitResult(128, "", "fatal: 'agent/test-run-bad-worktree' already exists"),
    ):
        with pytest.raises(DockerRunnerError, match="already exists"):
            runner.launch_run(run_id, workflow_yaml, secrets_env)


# ---------------------------------------------------------------------------
# list_active_runs
# ---------------------------------------------------------------------------


def test_list_active_runs_filters_and_reconstructs(runner: DockerRunner, repo_path: Path) -> None:
    ps_output = "deadbeef1234\tbernstein-run-a\nfeedface5678\tbernstein-run-b\n"
    inspect_a = {
        "NetworkSettings": {"Ports": {"18000/tcp": [{"HostPort": "18000"}]}},
        "Config": {"Image": "bernstein:latest"},
        "State": {"StartedAt": "2026-07-04T00:00:00Z"},
    }
    inspect_b = {
        "NetworkSettings": {"Ports": {"18001/tcp": [{"HostPort": "18001"}]}},
        "Config": {"Image": "bernstein:latest"},
        "State": {"StartedAt": "2026-07-04T00:01:00Z"},
    }

    call_sequence = [
        _completed(0, stdout=json.dumps(inspect_a)),
        _completed(0, stdout=json.dumps(inspect_b)),
    ]

    def fake_run(args, **kwargs):
        return call_sequence.pop(0)

    with (
        patch("bernstein.core.docker_runner._run_subprocess", side_effect=fake_run),
        patch("bernstein.core.docker_runner._run_or_raise", side_effect=lambda args, **kw: _completed(0, stdout=ps_output)),
    ):
        infos = runner.list_active_runs()

    assert len(infos) == 2
    assert {i.run_id for i in infos} == {"run-a", "run-b"}
    assert {i.port for i in infos} == {18000, 18001}
    for info in infos:
        assert info.container_name.startswith("bernstein-")
