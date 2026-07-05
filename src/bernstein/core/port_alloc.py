"""Port-range auto-assignment for parallel Bernstein runs.

Bernstein historically bound the task server to a single hardcoded default
port (8052) and wrote all runtime state (PID files, logs, signal files) to a
flat ``.sdd/runtime/`` directory. That made it impossible to run two
Bernstein workflows concurrently against the same repo -- the second
instance's singleton PID-lock guard would refuse to start.

This module resolves an actual TCP port to bind to, either:

* the caller-supplied ``--port`` (explicit, unchanged default-case behaviour
  -- still 8052 when nothing else is requested), or
* an auto-assigned free port from a configurable range (default
  ``18000-18999``, matching the existing convention already used by
  :mod:`bernstein.core.docker_runner` for per-run Docker port allocation)
  when the caller opts in via ``--auto-port`` / ``auto_port=True``.

Callers then pass the resolved port into :func:`bernstein.core.persistence
.runtime_state.get_runtime_dir` to namespace all per-instance runtime state
under ``.sdd/runtime/<port>/`` instead of the old flat layout.
"""

from __future__ import annotations

import logging
import os
import socket
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Matches the port-range convention already used by DockerRunner
# (bernstein/core/docker_runner.py) for per-run container port allocation.
DEFAULT_PORT_RANGE: tuple[int, int] = (18000, 18999)

BERNSTEIN_PORT_RANGE_ENV = "BERNSTEIN_PORT_RANGE"
"""Env var override for the auto-assignment range, format ``"18000-18999"``.

Follows the same ``os.environ.get("BERNSTEIN_...")`` convention used
throughout the CLI (see ``bernstein.cli.helpers.SERVER_URL`` /
``BERNSTEIN_AUTH_TOKEN`` for precedent).
"""

DEFAULT_BERNSTEIN_PORT = 8052
"""Single-run default port. Unchanged from historical behaviour: a run with
no ``--port`` and no ``--auto-port`` still binds here."""


class PortRangeError(ValueError):
    """Raised when a port-range string or config value cannot be parsed."""


def parse_port_range(value: str) -> tuple[int, int]:
    """Parse a ``"<low>-<high>"`` string into an inclusive ``(low, high)`` tuple.

    Args:
        value: Range string, e.g. ``"18000-18999"``.

    Returns:
        ``(low, high)`` with ``low <= high``.

    Raises:
        PortRangeError: If *value* is not a well-formed ``low-high`` range,
            either bound is out of the valid TCP port range (1-65535), or
            ``low > high``.
    """
    parts = value.strip().split("-")
    if len(parts) != 2:
        raise PortRangeError(f"Expected '<low>-<high>' (e.g. '18000-18999'), got: {value!r}")
    try:
        low, high = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise PortRangeError(f"Non-integer bound in port range {value!r}") from exc
    if not (1 <= low <= 65535 and 1 <= high <= 65535):
        raise PortRangeError(f"Port range {value!r} out of valid TCP port bounds (1-65535)")
    if low > high:
        raise PortRangeError(f"Port range {value!r} has low > high")
    return (low, high)


def _read_yaml_port_range(seed_path: Path | None) -> tuple[int, int] | None:
    """Best-effort read of the top-level ``port_range:`` key from *seed_path*.

    Deliberately does NOT go through the full :class:`SeedConfig` schema --
    ``port_range`` is a CLI/runtime-launch concern, not a task-authoring
    concern, so this is a narrow, self-contained YAML peek rather than a
    change to the seed dataclass + parser surface.
    """
    if seed_path is None or not seed_path.exists():
        return None
    try:
        import yaml

        data = yaml.safe_load(seed_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("port_alloc: could not read %s for port_range: %s", seed_path, exc)
        return None
    if not isinstance(data, dict):
        return None
    raw = data.get("port_range")
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        logger.warning(
            "port_alloc: bernstein.yaml port_range must be a 2-item list [low, high], got: %r -- ignoring",
            raw,
        )
        return None
    try:
        low, high = int(raw[0]), int(raw[1])
    except (TypeError, ValueError):
        logger.warning("port_alloc: bernstein.yaml port_range has non-integer bounds: %r -- ignoring", raw)
        return None
    if low > high:
        logger.warning("port_alloc: bernstein.yaml port_range %r has low > high -- ignoring", raw)
        return None
    return (low, high)


def resolve_port_range(seed_path: Path | None = None) -> tuple[int, int]:
    """Resolve the auto-assignment port range.

    Precedence (highest first), matching the codebase's existing
    ``BERNSTEIN_*`` env-var-overrides-config convention:

    1. ``BERNSTEIN_PORT_RANGE`` env var, format ``"18000-18999"``.
    2. ``port_range: [18000, 18999]`` key in ``bernstein.yaml`` (*seed_path*).
    3. :data:`DEFAULT_PORT_RANGE` (``18000-18999``).

    Args:
        seed_path: Path to the project's ``bernstein.yaml``, if known.

    Returns:
        Inclusive ``(low, high)`` port range.
    """
    env_val = os.environ.get(BERNSTEIN_PORT_RANGE_ENV, "").strip()
    if env_val:
        try:
            parsed = parse_port_range(env_val)
            logger.info(
                "resolve_port_range: using %s=%r -> range %s",
                BERNSTEIN_PORT_RANGE_ENV,
                env_val,
                parsed,
            )
            return parsed
        except PortRangeError as exc:
            logger.warning(
                "resolve_port_range: invalid %s=%r (%s) -- falling back to bernstein.yaml/default",
                BERNSTEIN_PORT_RANGE_ENV,
                env_val,
                exc,
            )

    yaml_range = _read_yaml_port_range(seed_path)
    if yaml_range is not None:
        logger.info("resolve_port_range: using bernstein.yaml port_range -> range %s", yaml_range)
        return yaml_range

    logger.debug("resolve_port_range: no override found -- using default range %s", DEFAULT_PORT_RANGE)
    return DEFAULT_PORT_RANGE


def is_port_free(port: int, host: str = "127.0.0.1") -> bool:
    """Return True if a TCP socket can bind to ``(host, port)`` right now.

    Uses a real bind-and-close probe (with ``SO_REUSEADDR``) rather than
    shelling out to ``lsof``/``netstat`` for portability across platforms --
    the codebase's own port-holder killer (``stop_cmd._find_port_pids_unix``)
    already shells out to ``lsof`` for a *different* purpose (finding which
    PID owns a bound port to kill it), but a plain free/bound check doesn't
    need a subprocess.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def find_free_port(
    port_range: tuple[int, int] | None = None,
    *,
    host: str = "127.0.0.1",
    seed_path: Path | None = None,
) -> int:
    """Find a free TCP port in *port_range* (or the resolved default range).

    Args:
        port_range: Explicit ``(low, high)`` range to scan. When ``None``,
            resolved via :func:`resolve_port_range` (env var > bernstein.yaml
            > default).
        host: Interface to probe against (default ``127.0.0.1``, matching
            the server's own default bind host).
        seed_path: Forwarded to :func:`resolve_port_range` when *port_range*
            is not given.

    Returns:
        The first free port found, scanning low-to-high.

    Raises:
        RuntimeError: If every port in the range is bound.
    """
    low, high = port_range if port_range is not None else resolve_port_range(seed_path)
    logger.info("find_free_port: scanning range %d-%d on %s for a free port", low, high, host)
    checked: list[int] = []
    for port in range(low, high + 1):
        checked.append(port)
        if is_port_free(port, host=host):
            logger.info(
                "find_free_port: selected port %d (checked %d port(s) in range %d-%d)",
                port,
                len(checked),
                low,
                high,
            )
            return port
        logger.debug("find_free_port: port %d in use, trying next", port)

    raise RuntimeError(f"Port range {low}-{high} exhausted: every port is in use on {host}")


def resolve_launch_port(
    *,
    requested_port: int | None,
    auto_port: bool,
    seed_path: Path | None = None,
    host: str = "127.0.0.1",
) -> int:
    """Resolve the actual port a new Bernstein instance should bind to.

    This is the single decision point CLI entry points call after parsing
    ``--port``/``--auto-port``. Logs the decision and why it was made.

    Args:
        requested_port: The ``--port`` value as parsed by Click (its own
            default is :data:`DEFAULT_BERNSTEIN_PORT`, so this is rarely
            ``None`` in practice; ``None`` is treated the same as the
            default port).
        auto_port: Whether ``--auto-port`` (or ``auto_port: true`` in
            bernstein.yaml) was requested.
        seed_path: Project's ``bernstein.yaml``, for range resolution.
        host: Bind host to probe against.

    Returns:
        The port to actually launch on.
    """
    base_port = requested_port if requested_port is not None else DEFAULT_BERNSTEIN_PORT

    if not auto_port:
        logger.info(
            "resolve_launch_port: auto_port not requested -- using requested/default port %d unchanged",
            base_port,
        )
        return base_port

    # --auto-port was requested. If the caller also passed an explicit
    # non-default --port, honour it as a starting preference: try it first,
    # then fall back into the auto-assign range.
    if (
        requested_port is not None
        and requested_port != DEFAULT_BERNSTEIN_PORT
        and is_port_free(requested_port, host=host)
    ):
        logger.info(
            "resolve_launch_port: auto_port requested but explicit --port %d is free -- using it",
            requested_port,
        )
        return requested_port

    port_range = resolve_port_range(seed_path)
    logger.info(
        "resolve_launch_port: auto_port requested -- assigning from range %d-%d",
        port_range[0],
        port_range[1],
    )
    return find_free_port(port_range, host=host, seed_path=seed_path)
