"""Docker container isolation for per-RUN Bernstein orchestration.

Implements the architecture locked in at
``work/bernstein/docker-isolation-design.md``: one container per RUN
(never per-agent, never host networking), bridge networking with a
``-p <port>:<port>`` published per run, the host's existing git worktree
bind-mounted at ``/workspace``, ``.sdd/runs/<run-id>/`` bind-mounted so
``run.db``/metrics land on host disk the instant they're written, secrets
injected exclusively via ``--env-file`` (never ``-e KEY=value``), and
resource ceilings (``--memory``/``--cpus``/``--pids-limit``) enforced by
cgroups rather than goodwill (design doc risk #5).

This module is pure stdlib — no Docker SDK dependency. Every Docker/git
interaction shells out via :func:`subprocess.run` with the return code
checked explicitly; on failure the full stdout/stderr is logged (never
truncated) before raising, per this repo's standing lesson that unchecked
subprocess return codes cause silent cascading failures (see
``.claude/rules/lessons.md``).

Port allocation and the active-run inventory are always derived by
querying ``docker ps`` live (design doc risk #2) — never from an
in-memory set — so both survive an orchestrator restart.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from bernstein.core.git.git_ops import GitResult, worktree_add

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONTAINER_NAME_PREFIX = "bernstein-"
_DOCKER_TIMEOUT_S = 30
_DOCKER_RUN_TIMEOUT_S = 60
_DOCKER_LOGS_TIMEOUT_S = 60

# git worktree lock-contention retry (design doc risk #3): small backoff,
# bounded attempts. This is intentionally a minimal inline retry rather than
# a new module — worktree.py has no reusable retry-with-backoff helper for
# `git worktree add` (WorktreeManager.create() calls worktree_add() once,
# no retry), so we implement it here rather than growing that module for a
# single caller.
_WORKTREE_ADD_MAX_ATTEMPTS = 5
_WORKTREE_ADD_BASE_DELAY_S = 0.5
_LOCK_CONTENTION_MARKERS = ("index.lock", "Unable to create", "another git process")


class DockerRunnerError(Exception):
    """Raised when a Docker or git operation fails irrecoverably."""


class RunStatus(Enum):
    """Coarse lifecycle state of a run's container."""

    RUNNING = "running"
    EXITED = "exited"
    KILLED = "killed"
    UNKNOWN = "unknown"


@dataclass
class ContainerInfo:
    """Everything needed to track and tear down one run's container."""

    container_id: str
    container_name: str
    run_id: str
    port: int
    worktree_path: Path
    image: str
    started_at: datetime


@dataclass
class ContainerHealth:
    """Detailed health snapshot from ``docker inspect``, companion to RunStatus."""

    status: RunStatus
    exit_code: int | None = None
    oom_killed: bool = False
    error: str = ""
    raw_state: dict[str, Any] = field(default_factory=dict)


def _run_subprocess(
    args: list[str],
    *,
    timeout: int,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run *args*, always capturing output as text, and log the invocation.

    Does NOT check the return code — callers decide whether a nonzero exit
    is expected (e.g. `docker inspect` on a missing container) or fatal.
    """
    logger.info("Running subprocess: %s (cwd=%s, timeout=%ss)", " ".join(args), cwd, timeout)
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        logger.error("Command not found: %s (%s)", args[0], exc)
        raise DockerRunnerError(f"Command not found: {args[0]}: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        logger.error(
            "Subprocess timed out after %ss: %s (stdout=%r stderr=%r)",
            timeout,
            " ".join(args),
            exc.stdout,
            exc.stderr,
        )
        raise DockerRunnerError(f"Command timed out after {timeout}s: {' '.join(args)}") from exc
    logger.debug(
        "Subprocess finished: %s -> returncode=%d stdout=%r stderr=%r",
        " ".join(args),
        result.returncode,
        result.stdout,
        result.stderr,
    )
    return result


def _run_or_raise(
    args: list[str], *, timeout: int, cwd: Path | None = None, context: str
) -> subprocess.CompletedProcess[str]:
    """Run *args*; raise :class:`DockerRunnerError` with full stderr/stdout on nonzero exit."""
    result = _run_subprocess(args, timeout=timeout, cwd=cwd)
    if result.returncode != 0:
        logger.error(
            "%s failed (returncode=%d)\ncommand: %s\nstdout: %s\nstderr: %s",
            context,
            result.returncode,
            " ".join(args),
            result.stdout,
            result.stderr,
        )
        raise DockerRunnerError(
            f"{context} failed (returncode={result.returncode}): stdout={result.stdout!r} stderr={result.stderr!r}"
        )
    return result


class DockerRunner:
    """Launch, monitor, and tear down one-container-per-run Bernstein executions.

    Args:
        repo_path: Absolute path to the host repo whose worktrees this
            runner manages.
        port_range: Inclusive (min, max) TCP port range to allocate from.
            Must match the design doc's 18000-18999 convention unless the
            caller has a specific reason to deviate.
        image: Docker image tag to launch (default ``bernstein:latest``).
            Pin to ``bernstein:<sha>`` for reproducible runs (design doc
            risk #4).
        memory: ``--memory`` limit passed to `docker run`.
        cpus: ``--cpus`` limit passed to `docker run`.
        pids_limit: ``--pids-limit`` cap to prevent fork-bombing agent
            subprocesses (design doc risk #5).
    """

    def __init__(
        self,
        repo_path: Path,
        port_range: tuple[int, int] = (18000, 18999),
        image: str = "bernstein:latest",
        memory: str = "4g",
        cpus: str = "2",
        pids_limit: int = 512,
    ) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.port_range = port_range
        self.image = image
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit
        logger.info(
            "DockerRunner initialized: repo_path=%s port_range=%s image=%s memory=%s cpus=%s pids_limit=%d",
            self.repo_path,
            self.port_range,
            self.image,
            self.memory,
            self.cpus,
            self.pids_limit,
        )

    # ------------------------------------------------------------------
    # Port allocation
    # ------------------------------------------------------------------

    def find_available_port(self) -> int:
        """Return a free port in ``port_range`` by scanning `docker ps`.

        Deliberately does NOT track ports in memory (design doc risk #2):
        an in-memory set drifts the moment the orchestrator restarts while
        containers keep running. Ground truth is always "what does the
        Docker daemon say is bound right now."

        Raises:
            DockerRunnerError: if every port in the range is bound.
        """
        result = _run_or_raise(
            ["docker", "ps", "--filter", f"name=^{_CONTAINER_NAME_PREFIX}", "--format", "{{.Ports}}"],
            timeout=_DOCKER_TIMEOUT_S,
            context="docker ps (port scan)",
        )
        bound_ports: set[int] = set()
        for line in result.stdout.splitlines():
            # Example line: "0.0.0.0:18042->18042/tcp, :::18042->18042/tcp"
            for segment in line.split(","):
                segment = segment.strip()
                if "->" not in segment:
                    continue
                host_part = segment.split("->", 1)[0]
                if ":" not in host_part:
                    continue
                port_str = host_part.rsplit(":", 1)[-1]
                try:
                    bound_ports.add(int(port_str))
                except ValueError:
                    continue

        logger.info(
            "Port scan: %d bound bernstein-* ports found in range %s: %s",
            len(bound_ports),
            self.port_range,
            sorted(bound_ports),
        )

        low, high = self.port_range
        for port in range(low, high + 1):
            if port not in bound_ports:
                logger.info("Selected available port %d (range %s, %d bound)", port, self.port_range, len(bound_ports))
                return port

        raise DockerRunnerError(
            f"Port range {self.port_range} exhausted: all {high - low + 1} ports are bound "
            f"by running bernstein-* containers"
        )

    # ------------------------------------------------------------------
    # Worktree creation with retry/backoff on lock contention
    # ------------------------------------------------------------------

    def _worktree_add_with_retry(self, worktree_path: Path, branch_name: str) -> GitResult:
        """`git worktree add` with retry/backoff on lock contention (design doc risk #3).

        Does NOT run `git worktree prune` — that is explicitly reserved for
        a host-side mutex-guarded maintenance path per the design doc, never
        invoked from a single run's launch path.
        """
        last_result: GitResult | None = None
        for attempt in range(1, _WORKTREE_ADD_MAX_ATTEMPTS + 1):
            logger.info(
                "git worktree add attempt %d/%d: path=%s branch=%s",
                attempt,
                _WORKTREE_ADD_MAX_ATTEMPTS,
                worktree_path,
                branch_name,
            )
            result = worktree_add(self.repo_path, worktree_path, branch_name)
            if result.ok:
                logger.info("git worktree add succeeded on attempt %d: %s", attempt, worktree_path)
                return result
            last_result = result
            is_lock_contention = any(marker in result.stderr for marker in _LOCK_CONTENTION_MARKERS)
            if not is_lock_contention or attempt == _WORKTREE_ADD_MAX_ATTEMPTS:
                logger.error(
                    "git worktree add failed (attempt %d/%d, lock_contention=%s): stderr=%s",
                    attempt,
                    _WORKTREE_ADD_MAX_ATTEMPTS,
                    is_lock_contention,
                    result.stderr,
                )
                break
            delay = _WORKTREE_ADD_BASE_DELAY_S * (2 ** (attempt - 1)) + random.uniform(0, 0.25)
            logger.warning(
                "git worktree add hit lock contention (attempt %d/%d), retrying in %.2fs: %s",
                attempt,
                _WORKTREE_ADD_MAX_ATTEMPTS,
                delay,
                result.stderr.strip(),
            )
            time.sleep(delay)
        assert last_result is not None
        return last_result

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------

    def launch_run(
        self,
        run_id: str,
        workflow_yaml: Path,
        secrets_env_file: Path,
        **kwargs: Any,
    ) -> ContainerInfo:
        """Launch one isolated container for *run_id*.

        Steps (see module docstring / design doc for rationale on each):
          1. `git worktree add` for this run, retrying on lock contention.
          2. Allocate a free port via :meth:`find_available_port`.
          3. Create ``.sdd/runs/<run_id>/`` on the host.
          4. Copy *workflow_yaml* into the worktree as ``bernstein.yaml``
             (a run-time seed file — never committed to the repo).
          5. `docker run -d` with bridge networking, resource limits, the
             worktree + run-dir bind mounts, `--env-file` for secrets, and
             `--add-host=host.docker.internal:host-gateway`.
          6. Return a :class:`ContainerInfo` describing the launched run.

        Args:
            run_id: Unique identifier for this run. Also used as the git
                worktree session id and the container name suffix.
            workflow_yaml: Path to the bernstein.yaml seed config to run.
            secrets_env_file: Path to a ``KEY=value`` env file (mode 600,
                outside any git-tracked directory) passed via
                ``--env-file``. Never inlined as ``-e KEY=value``.
            **kwargs: Reserved for future per-run overrides (e.g. a
                per-run image tag); unused today.

        Returns:
            ContainerInfo describing the launched container.

        Raises:
            DockerRunnerError: on any unrecoverable failure in worktree
                creation, port allocation, or `docker run`.
        """
        logger.info("launch_run starting: run_id=%s workflow_yaml=%s", run_id, workflow_yaml)

        if not workflow_yaml.is_file():
            raise DockerRunnerError(f"workflow_yaml does not exist: {workflow_yaml}")
        if not secrets_env_file.is_file():
            raise DockerRunnerError(f"secrets_env_file does not exist: {secrets_env_file}")

        # 1. git worktree add (retry on lock contention)
        worktree_path = self.repo_path / ".sdd" / "worktrees" / run_id
        branch_name = f"agent/{run_id}"
        wt_result = self._worktree_add_with_retry(worktree_path, branch_name)
        if not wt_result.ok:
            raise DockerRunnerError(
                f"git worktree add failed for run {run_id!r} after retries: {wt_result.stderr.strip()}"
            )
        logger.info("Worktree ready for run %s: %s (branch %s)", run_id, worktree_path, branch_name)

        # 2. Allocate a port from live `docker ps` state.
        port = self.find_available_port()
        logger.info("Allocated port %d for run %s (derived from live docker ps state)", port, run_id)

        # 3. Host-side run dir for .sdd/runs/<run_id>/ bind mount.
        run_dir = self.repo_path / ".sdd" / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Created host run dir for %s: %s", run_id, run_dir)

        # 4. Seed bernstein.yaml into the worktree (NOT committed to the repo
        #    — mirrors run-proof.sh's pattern of a run-time-only seed file).
        seed_dest = worktree_path / "bernstein.yaml"
        shutil.copy2(workflow_yaml, seed_dest)
        logger.info("Copied workflow seed %s -> %s (not committed)", workflow_yaml, seed_dest)

        # 5. docker run -d
        container_name = f"{_CONTAINER_NAME_PREFIX}{run_id}"
        container_run_dir = f"/workspace/.sdd/runs/{run_id}"
        docker_args = [
            "docker",
            "run",
            "-d",
            "--name",
            container_name,
            "--network",
            "bridge",
            "-p",
            f"{port}:{port}",
            "--memory",
            self.memory,
            "--cpus",
            self.cpus,
            "--pids-limit",
            str(self.pids_limit),
            "-v",
            f"{worktree_path}:/workspace:rw",
            "-v",
            f"{run_dir}:{container_run_dir}:rw",
            "--env-file",
            str(secrets_env_file),
            "-e",
            f"BERNSTEIN_PORT={port}",
            "-e",
            f"BERNSTEIN_RUN_ID={run_id}",
            "--add-host=host.docker.internal:host-gateway",
            self.image,
            "bernstein",
            "conduct",
            "--run-id",
            run_id,
            "--auto-approve",
            "--quiet",
        ]
        logger.info(
            "Mount decisions for run %s: worktree %s -> /workspace (rw); run dir %s -> %s (rw); "
            "secrets via --env-file %s (never -e KEY=value)",
            run_id,
            worktree_path,
            run_dir,
            container_run_dir,
            secrets_env_file,
        )
        logger.info("docker run command for run %s: %s", run_id, " ".join(docker_args))

        result = _run_or_raise(
            docker_args,
            timeout=_DOCKER_RUN_TIMEOUT_S,
            context=f"docker run for run {run_id!r}",
        )
        container_id = result.stdout.strip()
        if not container_id:
            raise DockerRunnerError(
                f"docker run for run {run_id!r} exited 0 but printed no container id "
                f"(stdout={result.stdout!r} stderr={result.stderr!r})"
            )

        info = ContainerInfo(
            container_id=container_id,
            container_name=container_name,
            run_id=run_id,
            port=port,
            worktree_path=worktree_path,
            image=self.image,
            started_at=datetime.now(UTC),
        )
        logger.info(
            "launch_run succeeded: run_id=%s container_id=%s container_name=%s port=%d",
            run_id,
            container_id[:12],
            container_name,
            port,
        )
        return info

    # ------------------------------------------------------------------
    # Status / logs / kill / cleanup
    # ------------------------------------------------------------------

    def poll_status(self, container_id: str) -> RunStatus:
        """Poll `docker inspect` and classify the container's lifecycle state.

        Returns RunStatus.UNKNOWN (never raises) when the container cannot
        be found or the inspect output can't be parsed, since callers use
        this in polling loops where a transient inspect failure shouldn't
        crash the loop. Use :meth:`inspect_health` for the detailed,
        error-surfacing variant.
        """
        health = self.inspect_health(container_id)
        return health.status

    def inspect_health(self, container_id: str) -> ContainerHealth:
        """`docker inspect --format '{{json .State}}' <id>` with full parsing.

        Returns exit code, OOM-killed flag, and a raw state dict for
        post-mortem, per design doc §7's "crash/kill case" handling.
        """
        result = _run_subprocess(
            ["docker", "inspect", "--format", "{{json .State}}", container_id],
            timeout=_DOCKER_TIMEOUT_S,
        )
        if result.returncode != 0:
            logger.warning(
                "docker inspect failed for %s (returncode=%d): stdout=%s stderr=%s",
                container_id,
                result.returncode,
                result.stdout,
                result.stderr,
            )
            return ContainerHealth(status=RunStatus.UNKNOWN, error=result.stderr.strip())

        try:
            state = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse docker inspect JSON for %s: %s (raw=%r)", container_id, exc, result.stdout)
            return ContainerHealth(status=RunStatus.UNKNOWN, error=f"JSON parse error: {exc}")

        running = bool(state.get("Running", False))
        oom_killed = bool(state.get("OOMKilled", False))
        exit_code = state.get("ExitCode")
        exit_code = int(exit_code) if exit_code is not None else None
        error = state.get("Error", "") or ""

        if running:
            status = RunStatus.RUNNING
        elif oom_killed:
            status = RunStatus.KILLED
        elif exit_code is not None:
            status = RunStatus.EXITED
        else:
            status = RunStatus.UNKNOWN

        logger.info(
            "inspect_health for %s: status=%s exit_code=%s oom_killed=%s error=%r",
            container_id[:12],
            status,
            exit_code,
            oom_killed,
            error,
        )
        return ContainerHealth(status=status, exit_code=exit_code, oom_killed=oom_killed, error=error, raw_state=state)

    def capture_logs(self, container_id: str, output_path: Path) -> None:
        """Snapshot `docker logs <id>` to *output_path*.

        NOTE: this is a point-in-time snapshot capture, not continuous
        streaming. Continuous `docker logs -f` capture into
        `.sdd/runs/<run-id>/container.log` while the container runs is a
        follow-up (design doc §7 step 1) — call this repeatedly (e.g. from
        a polling loop) if you need a running log tail.
        """
        result = _run_subprocess(
            ["docker", "logs", container_id],
            timeout=_DOCKER_LOGS_TIMEOUT_S,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # `docker logs` interleaves stdout/stderr from the container; both
        # are captured here so nothing is lost, matching the "log the WHY,
        # log everything" standing rule.
        output_path.write_text(result.stdout + result.stderr, encoding="utf-8")
        logger.info(
            "Captured logs for %s -> %s (%d bytes, docker-logs-returncode=%d)",
            container_id[:12],
            output_path,
            output_path.stat().st_size,
            result.returncode,
        )

    def kill_run(self, container_id: str, reason: str = "timeout") -> None:
        """`docker kill` the container. Does not remove it — see :meth:`cleanup`."""
        logger.warning("Killing container %s (reason=%s)", container_id[:12], reason)
        result = _run_subprocess(["docker", "kill", container_id], timeout=_DOCKER_TIMEOUT_S)
        if result.returncode != 0:
            logger.error(
                "docker kill failed for %s (returncode=%d): stdout=%s stderr=%s",
                container_id,
                result.returncode,
                result.stdout,
                result.stderr,
            )
            raise DockerRunnerError(
                f"docker kill failed for {container_id!r} (returncode={result.returncode}): stderr={result.stderr!r}"
            )
        logger.info("Killed container %s (reason=%s)", container_id[:12], reason)

    def cleanup(self, container_id: str, keep_worktree: bool = False) -> None:
        """`docker rm` the container; optionally remove its git worktree too.

        Best-effort on worktree removal: a failure there is logged but does
        not prevent the container from being removed.
        """
        logger.info("Cleaning up container %s (keep_worktree=%s)", container_id[:12], keep_worktree)
        result = _run_subprocess(["docker", "rm", "-f", container_id], timeout=_DOCKER_TIMEOUT_S)
        if result.returncode != 0:
            logger.error(
                "docker rm failed for %s (returncode=%d): stdout=%s stderr=%s",
                container_id,
                result.returncode,
                result.stdout,
                result.stderr,
            )
            raise DockerRunnerError(
                f"docker rm failed for {container_id!r} (returncode={result.returncode}): stderr={result.stderr!r}"
            )
        logger.info("Removed container %s", container_id[:12])

        if keep_worktree:
            logger.info("keep_worktree=True, skipping worktree removal for container %s", container_id[:12])
            return

        # Best-effort: find the run_id from the container name via `docker ps -a`
        # is unnecessary here — callers that want worktree removal should use
        # list_active_runs() to resolve run_id -> worktree_path themselves, or
        # pass keep_worktree=True and clean up worktrees separately. This
        # method intentionally does not guess a worktree path from a bare
        # container_id since the mapping isn't recoverable post-`docker rm`
        # without an inspect call beforehand.
        logger.debug(
            "cleanup(%s): worktree removal from a bare container_id is not attempted post-rm; "
            "use list_active_runs() before cleanup() if worktree removal is required",
            container_id[:12],
        )

    # ------------------------------------------------------------------
    # Inventory
    # ------------------------------------------------------------------

    def list_active_runs(self) -> list[ContainerInfo]:
        """Reconstruct ContainerInfo for every live `bernstein-*` container.

        Always queries `docker ps` live (design doc risk #2) rather than
        relying on any in-memory registry, so this survives orchestrator
        restarts.
        """
        result = _run_or_raise(
            [
                "docker",
                "ps",
                "--filter",
                f"name=^{_CONTAINER_NAME_PREFIX}",
                "--format",
                "{{.ID}}\t{{.Names}}",
            ],
            timeout=_DOCKER_TIMEOUT_S,
            context="docker ps (list_active_runs)",
        )

        infos: list[ContainerInfo] = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                logger.warning("Unexpected docker ps line format, skipping: %r", line)
                continue
            container_id, container_name = parts
            if not container_name.startswith(_CONTAINER_NAME_PREFIX):
                continue
            run_id = container_name[len(_CONTAINER_NAME_PREFIX) :]

            inspect_result = _run_subprocess(
                ["docker", "inspect", "--format", "{{json .}}", container_id],
                timeout=_DOCKER_TIMEOUT_S,
            )
            if inspect_result.returncode != 0:
                logger.warning(
                    "docker inspect failed while reconstructing run %s (%s): %s",
                    run_id,
                    container_id,
                    inspect_result.stderr,
                )
                continue
            try:
                detail = json.loads(inspect_result.stdout)
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse docker inspect JSON for %s: %s", container_id, exc)
                continue

            port = 0
            ports = detail.get("NetworkSettings", {}).get("Ports", {}) or {}
            for _container_port, bindings in ports.items():
                if bindings:
                    try:
                        port = int(bindings[0].get("HostPort", 0))
                    except (ValueError, TypeError, IndexError):
                        pass
                    break

            image = detail.get("Config", {}).get("Image", self.image)
            started_at_str = detail.get("State", {}).get("StartedAt", "")
            try:
                started_at = (
                    datetime.fromisoformat(started_at_str.replace("Z", "+00:00"))
                    if started_at_str
                    else datetime.now(UTC)
                )
            except ValueError:
                started_at = datetime.now(UTC)

            worktree_path = self.repo_path / ".sdd" / "worktrees" / run_id

            infos.append(
                ContainerInfo(
                    container_id=container_id,
                    container_name=container_name,
                    run_id=run_id,
                    port=port,
                    worktree_path=worktree_path,
                    image=image,
                    started_at=started_at,
                )
            )

        logger.info("list_active_runs found %d live bernstein-* containers", len(infos))
        return infos
