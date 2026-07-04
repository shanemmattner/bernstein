"""Wave-3 per-agent instrumentation: LLM calls, tool calls, and conversation history.

Wave 2 (see ``.sdd/runs/<run_id>/summary.json`` and
:mod:`bernstein.core.orchestration.run_report`) added phase/task-level timing
that the top-level orchestrator can see directly. This module adds the layer
below it: individual LLM API calls, individual tool invocations, and the
growing message history *inside* a single agent process - things the
orchestrator cannot see because they happen inside an agent subprocess (or,
for in-process runners, inside a single call to an SDK).

This is a strictly additive, observe-only instrumentation layer. Every public
method is wrapped in a broad ``try/except`` and degrades to a logged warning
on any failure - instrumentation must NEVER crash or block the agent run it
is observing (see ``.claude/rules/lessons.md`` rule 2: "logging IS the
debugging interface", and the corollary that observability code itself must
be maximally defensive).

Pluggable backends (see ``work/bernstein/sqlite-run-storage-design.md``):
``RunInstrumenter`` is a thin facade that computes/normalizes derived fields
(``wall_ms``, truncation, defaulted ``total_tokens``) once, then delegates the
actual write to an :class:`InstrumenterBackend`. Two backends ship today:

* :class:`JSONLInstrumenterBackend` - the original file-per-stream format,
  unchanged byte-for-byte from the pre-refactor implementation::

      .sdd/runs/<run_id>/tasks/<task_id>/agents/<agent_id>/llm-calls.jsonl
      .sdd/runs/<run_id>/tasks/<task_id>/agents/<agent_id>/tool-calls.jsonl
      .sdd/runs/<run_id>/tasks/<task_id>/agents/<agent_id>/conversation.jsonl

* :class:`SQLiteInstrumenterBackend` - a single ``run.db`` (WAL mode,
  foreign keys on) written under the same ``base_dir`` the caller already
  resolves (today: ``.../agents/<agent_id>/run.db``). Selected via
  ``init_instrumenter(..., backend="sqlite")`` - the default.

Design choice - module-level singleton, not a class threaded through every
call site: the two hook points wired up in wave 3
(:mod:`bernstein.adapters.openai_agents_runner` and
:mod:`bernstein.adapters.openai_agents_builtins`) are built from free
functions and module-level state (``emit_event``, the builtin tool
closures), not a class with a natural place to stash ``self``. Each of those
processes is already a dedicated one-agent-per-process runner (the adapter
spawns ``python -m bernstein.adapters.openai_agents_runner`` once per agent
session), so "one instance per agent process" and "one module-level
singleton" are the same thing here. ``init_instrumenter``/``get_instrumenter``
avoid plumbing an instrumenter parameter through a dozen existing function
signatures. A caller that needs multiple independent instrumenters in one
process (e.g. a test) can still construct :class:`RunInstrumenter` directly
and never touch the singleton.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from bernstein.core.security.sanitize import sanitize_log

logger = logging.getLogger(__name__)

# Env var carrying the orchestrator's run id down into agent subprocesses.
# Must be added to the adapter env-isolation allowlist
# (``bernstein.adapters.env_isolation._BASE_ALLOWLIST``) or a filtered
# subprocess env silently drops it and every agent falls back to "unknown".
RUN_ID_ENV_VAR = "BERNSTEIN_RUN_ID"

# Truncation cap for individual string values inside logged tool-call args.
# Prevents a single huge string argument (e.g. a full file body passed to
# write_file) from bloating tool-calls.jsonl - only metadata/preview is
# wanted here, never full payloads (see module docstring + task spec).
_ARG_VALUE_TRUNCATE_CHARS = 500
_TRUNCATE_MARKER = "...[truncated]"

# Truncation cap for a single conversation message's ``content`` (bug fix,
# 2026-07-04: log_message() previously recorded only content_length, never
# the actual text, making conversation.jsonl useless for debugging what an
# agent actually said - see commit b38fc3fe). 2000 chars is enough to see
# the shape/gist of a message without conversation.jsonl growing unbounded
# on a long run.
_MESSAGE_CONTENT_TRUNCATE_CHARS = 2000

# Truncation cap for a tool call's ``result`` (bug fix, 2026-07-04:
# log_tool_call() previously recorded only name/args/success, never what the
# tool actually returned, making it impossible to see tool output without
# re-running the agent - see commit b38fc3fe). 1000 chars is enough for a
# preview of most tool outputs (file reads, command output, etc.) without
# duplicating huge payloads into tool-calls.jsonl.
_TOOL_RESULT_TRUNCATE_CHARS = 1000

# Filesystem-safe shape for a single directory-name component. run_id arrives
# via an environment variable and task_id/agent_id via the runner manifest,
# so :func:`resolve_agent_dir` treats all three as untrusted before joining
# them into a path (see :func:`_sanitize_path_component`).
_SAFE_COMPONENT_RE = re.compile(r"[^A-Za-z0-9._-]")
_MAX_COMPONENT_CHARS = 128
_FALLBACK_COMPONENT = "unknown"

# Default backend for init_instrumenter() - see backend-selection log lines
# below for the "which backend did this process actually get" answer.
_DEFAULT_BACKEND = "sqlite"


def _sanitize_path_component(value: str) -> str:
    """Reduce an untrusted id to a single safe directory-name component.

    Keeps only the final path component (dropping separators, absolute
    prefixes, and ``..`` segments before it), replaces every character
    outside ``[A-Za-z0-9._-]`` with ``_``, caps the length, and collapses
    anything that could still escape the directory (empty string, ``.``,
    ``..``, all-dot names) to ``"unknown"``. Never raises - a hostile or
    malformed id degrades to a wrong-but-contained directory name, matching
    this module's observe-only contract.
    """
    cleaned = value.replace("\x00", "").strip()
    # Normalize backslashes so a Windows-style separator cannot survive as a
    # literal character on POSIX, then keep only the last path component.
    cleaned = PurePosixPath(cleaned.replace("\\", "/")).name
    cleaned = _SAFE_COMPONENT_RE.sub("_", cleaned)[:_MAX_COMPONENT_CHARS]
    if not cleaned or cleaned.strip(".") == "":
        return _FALLBACK_COMPONENT
    return cleaned


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with millisecond precision."""
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _truncate_value(value: Any, *, max_chars: int = _ARG_VALUE_TRUNCATE_CHARS) -> Any:
    """Recursively truncate long strings inside an args structure.

    Only strings are truncated; other JSON-safe scalar types pass through
    unchanged. Dicts and lists are walked recursively so nested large
    payloads (e.g. ``{"content": "<huge file body>"}``) are capped without
    losing the surrounding structure. Non-JSON-safe objects are stringified
    first so a stray non-serializable arg never breaks the write.
    """
    if isinstance(value, str):
        if len(value) > max_chars:
            return value[:max_chars] + _TRUNCATE_MARKER
        return value
    if isinstance(value, dict):
        return {str(k): _truncate_value(v, max_chars=max_chars) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_truncate_value(v, max_chars=max_chars) for v in value]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    # Fallback for anything else JSON can't natively encode (e.g. custom
    # objects some adapter passed through as a tool arg).
    try:
        text = str(value)
    except Exception:  # pragma: no cover - defensive, str() essentially never raises
        return "<unrepr-able value>"
    return _truncate_value(text, max_chars=max_chars)


def _truncate_text(value: Any, *, max_chars: int, field_name: str, key: str) -> str | None:
    """Best-effort stringify + truncate a single value for a flat text field.

    Unlike :func:`_truncate_value` (which recurses into dict/list structures
    for tool-call *args*), this is for a single flat text field - tool
    *results* and conversation message *content* - which may arrive as
    arbitrary objects (SDK response items, exceptions, etc.) and must always
    degrade to a string rather than raise. Returns ``None`` when ``value``
    is ``None`` (nothing to log) so callers can omit the key entirely rather
    than writing a misleading ``"content": "None"`` line/column.

    Logs at DEBUG whenever truncation actually happens, with the original
    and truncated lengths, per the "log every truncation" instrumentation
    rule. This is shared by every backend since truncation policy is a
    serialization-time concern, independent of the sink (file vs DB).
    """
    if value is None:
        return None
    try:
        text = value if isinstance(value, str) else str(value)
    except Exception as exc:  # pragma: no cover - defensive, str() essentially never raises
        logger.debug(
            "RunInstrumenter: failed to stringify %s for %s=%s: %s", field_name, field_name, sanitize_log(key), exc
        )
        return "<unrepr-able value>"
    if len(text) > max_chars:
        logger.debug(
            "RunInstrumenter: truncating %s for %s=%s from %d chars to %d chars",
            field_name,
            field_name,
            sanitize_log(key),
            len(text),
            max_chars,
        )
        return text[:max_chars] + _TRUNCATE_MARKER
    return text


def _iso_delta_ms(ts_start: str, ts_end: str) -> float | None:
    """Best-effort millisecond delta between two ISO-8601 timestamps.

    Returns ``None`` (never raises, never fabricates a value) when either
    timestamp fails to parse.
    """
    try:
        start = datetime.fromisoformat(ts_start)
        end = datetime.fromisoformat(ts_end)
        return (end - start).total_seconds() * 1000.0
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Pluggable backend interface
# ---------------------------------------------------------------------------


class InstrumenterBackend(ABC):
    """Sink for instrumentation records - one instance per (set of) target dir(s).

    ``RunInstrumenter`` (the facade) computes derived fields (``wall_ms``,
    defaulted ``total_tokens``, truncated args/result/content) once and
    delegates the actual write to a backend implementation. Every method
    must be safe to call at high frequency and must never raise -
    implementations are still expected to catch their own errors (SQLite
    exceptions, disk I/O errors) and log+swallow, matching the facade's own
    defensive posture, since a backend bug must not propagate up into the
    calling agent process either.
    """

    @abstractmethod
    def log_llm_call(
        self,
        *,
        call_id: str,
        ts_start: str,
        ts_end: str,
        wall_ms: float | None,
        model: str,
        endpoint: str | None,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
        status: str,
        error: str | None,
        ts_ttft: str | None = None,
        tokens_per_sec: float | None = None,
    ) -> None:
        """Record one LLM API call."""

    @abstractmethod
    def log_tool_call(
        self,
        *,
        call_id: str,
        ts_start: str,
        ts_end: str,
        wall_ms: float | None,
        tool: str,
        args: dict[str, Any],
        result: str | None,
        success: bool,
        error: str | None,
    ) -> None:
        """Record one tool invocation. ``args``/``result`` are already truncated."""

    @abstractmethod
    def log_message(
        self,
        *,
        idx: int,
        role: str,
        content: str | None,
        content_length: int,
        ts: str,
        tool_calls: list[str] | None = None,
    ) -> None:
        """Record one conversation message. ``content`` is already truncated."""

    @abstractmethod
    def close(self) -> None:
        """Flush/cleanup any open resources (file handles, DB connections)."""


class JSONLInstrumenterBackend(InstrumenterBackend):
    """Appends JSONL instrumentation records - the original, pre-refactor format.

    One directory per agent, created on first write; three files
    (``llm-calls.jsonl``, ``tool-calls.jsonl``, ``conversation.jsonl``)
    appended to with one ``write()`` call per record. Extracted verbatim
    from the original :class:`RunInstrumenter` implementation - same file
    names, same paths, same line format as before this refactor.

    ``extra_dirs`` (Bug fix, instrumentation audit bug 3): when this agent
    process is working a BATCH of tasks in one session (see
    ``RunnerManifest.task_ids`` in :mod:`bernstein.adapters.openai_agents_runner`),
    every JSONL record written to ``base_dir`` is ALSO fanned out to each
    directory in ``extra_dirs`` so every task in the batch gets a full copy
    of this agent's instrumentation, not just the primary task. Defaults to
    an empty list (no fan-out) for the common single-task case.
    """

    def __init__(
        self,
        *,
        run_id: str,
        task_id: str,
        agent_id: str,
        base_dir: Path,
        extra_dirs: list[Path] | None = None,
    ) -> None:
        self.run_id = run_id
        self.task_id = task_id
        self.agent_id = agent_id
        self.base_dir = Path(base_dir)
        self.extra_dirs = [Path(d) for d in (extra_dirs or [])]
        self._lock = threading.Lock()
        self._dir_ready = False
        self._ready_extra_dirs: list[Path] = []
        logger.debug(
            "JSONLInstrumenterBackend: initializing run_id=%s task_id=%s agent_id=%s base_dir=%s extra_dirs=%s",
            sanitize_log(self.run_id),
            sanitize_log(self.task_id),
            sanitize_log(self.agent_id),
            self.base_dir,
            self.extra_dirs,
        )
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            self._dir_ready = True
            logger.info(
                "JSONLInstrumenterBackend ready run_id=%s task_id=%s agent_id=%s -> "
                "llm_calls=%s tool_calls=%s conversation=%s",
                sanitize_log(self.run_id),
                sanitize_log(self.task_id),
                sanitize_log(self.agent_id),
                self._llm_calls_path(),
                self._tool_calls_path(),
                self._conversation_path(),
            )
        except OSError as exc:
            logger.warning(
                "JSONLInstrumenterBackend: failed to create instrumentation dir %s (run_id=%s "
                "task_id=%s agent_id=%s): %s - instrumentation for this agent is "
                "DISABLED, the agent run itself is unaffected",
                self.base_dir,
                sanitize_log(self.run_id),
                sanitize_log(self.task_id),
                sanitize_log(self.agent_id),
                exc,
            )
        for extra_dir in self.extra_dirs:
            if extra_dir == self.base_dir:
                continue
            try:
                extra_dir.mkdir(parents=True, exist_ok=True)
                self._ready_extra_dirs.append(extra_dir)
                logger.info(
                    "JSONLInstrumenterBackend: fanning out run_id=%s task_id=%s agent_id=%s to EXTRA "
                    "batch-mate dir %s (batched-task instrumentation fix)",
                    sanitize_log(self.run_id),
                    sanitize_log(self.task_id),
                    sanitize_log(self.agent_id),
                    extra_dir,
                )
            except OSError as exc:
                logger.warning(
                    "JSONLInstrumenterBackend: failed to create EXTRA instrumentation dir %s "
                    "(run_id=%s task_id=%s agent_id=%s): %s - this batch-mate task will "
                    "still have zero instrumentation",
                    extra_dir,
                    sanitize_log(self.run_id),
                    sanitize_log(self.task_id),
                    sanitize_log(self.agent_id),
                    exc,
                )
        if self.extra_dirs:
            logger.info(
                "JSONLInstrumenterBackend: batch fan-out enabled for run_id=%s task_id=%s agent_id=%s -> "
                "%d/%d extra dir(s) ready: %s",
                sanitize_log(self.run_id),
                sanitize_log(self.task_id),
                sanitize_log(self.agent_id),
                len(self._ready_extra_dirs),
                len(self.extra_dirs),
                self._ready_extra_dirs,
            )

    # -- path helpers ---------------------------------------------------

    def _all_base_dirs(self) -> list[Path]:
        if self._dir_ready:
            return [self.base_dir, *self._ready_extra_dirs]
        return list(self._ready_extra_dirs)

    def _llm_calls_path(self) -> Path:
        return self.base_dir / "llm-calls.jsonl"

    def _tool_calls_path(self) -> Path:
        return self.base_dir / "tool-calls.jsonl"

    def _conversation_path(self) -> Path:
        return self.base_dir / "conversation.jsonl"

    def _append_line(self, path: Path, record: dict[str, Any], *, kind: str, key: str) -> None:
        """Serialize *record* and append it as a single line, single write() call.

        Building the full line before calling ``write()`` once (rather than
        writing pieces incrementally) is what keeps concurrent writers from
        the same process from interleaving partial lines - see module
        docstring. A ``threading.Lock`` additionally protects against two
        threads in the SAME process racing on the SAME file (e.g. a
        heartbeat thread and the main thread); separate agent PROCESSES
        never share a base_dir so no cross-process lock is needed.

        ``path`` is expected to be ``self.base_dir / <filename>``; when this
        backend was constructed with ``extra_dirs`` (batched tasks, see the
        module/class docstring), the SAME line is additionally appended to
        each batch-mate's copy of ``path.name`` under its own ready extra
        dir, so every task in the batch ends up with a full, independent set
        of JSONL files instead of only the first task.
        """
        targets = [d / path.name for d in self._all_base_dirs()]
        if not targets:
            logger.debug(
                "JSONLInstrumenterBackend: dropping %s record %s=%s - no ready instrumentation dir "
                "(dir creation failed or backend uninitialized)",
                kind,
                kind,
                sanitize_log(key),
            )
            return
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "JSONLInstrumenterBackend: failed to serialize %s record %s=%s: %s",
                kind,
                kind,
                sanitize_log(key),
                exc,
            )
            return
        with self._lock:
            for target in targets:
                try:
                    with target.open("a", encoding="utf-8") as fh:
                        fh.write(line + "\n")
                    logger.debug(
                        "JSONLInstrumenterBackend: wrote %s record %s=%s to %s",
                        kind,
                        kind,
                        sanitize_log(key),
                        target,
                    )
                except OSError as exc:
                    logger.warning(
                        "JSONLInstrumenterBackend: failed to write %s record %s=%s to %s: %s",
                        kind,
                        kind,
                        sanitize_log(key),
                        target,
                        exc,
                    )

    # -- InstrumenterBackend interface -----------------------------------

    def log_llm_call(
        self,
        *,
        call_id: str,
        ts_start: str,
        ts_end: str,
        wall_ms: float | None,
        model: str,
        endpoint: str | None,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
        status: str,
        error: str | None,
        ts_ttft: str | None = None,
        tokens_per_sec: float | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "call_id": call_id,
            "ts_start": ts_start,
            "ts_end": ts_end,
            "wall_ms": wall_ms,
            "model": model,
            "endpoint": endpoint,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "status": status,
            "error": error,
        }
        if ts_ttft is not None:
            record["ts_ttft"] = ts_ttft
        if tokens_per_sec is not None:
            record["tokens_per_sec"] = tokens_per_sec
        self._append_line(self._llm_calls_path(), record, kind="llm_call", key=call_id)

    def log_tool_call(
        self,
        *,
        call_id: str,
        ts_start: str,
        ts_end: str,
        wall_ms: float | None,
        tool: str,
        args: dict[str, Any],
        result: str | None,
        success: bool,
        error: str | None,
    ) -> None:
        record: dict[str, Any] = {
            "call_id": call_id,
            "ts_start": ts_start,
            "ts_end": ts_end,
            "wall_ms": wall_ms,
            "tool": tool,
            "args": args,
            "success": success,
            "error": error,
        }
        if result is not None:
            record["result"] = result
        self._append_line(self._tool_calls_path(), record, kind="tool_call", key=call_id)

    def log_message(
        self,
        *,
        idx: int,
        role: str,
        content: str | None,
        content_length: int,
        ts: str,
        tool_calls: list[str] | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "idx": idx,
            "role": role,
            "content_length": content_length,
            "ts": ts,
        }
        if content is not None:
            record["content"] = content
        if tool_calls:
            record["tool_calls"] = list(tool_calls)
        self._append_line(self._conversation_path(), record, kind="message", key=str(idx))

    def close(self) -> None:
        # Nothing to flush - every write is already a synchronous, closed
        # file handle (opened/closed per append in _append_line).
        logger.debug(
            "JSONLInstrumenterBackend: close() run_id=%s task_id=%s agent_id=%s (no-op, files already flushed)",
            sanitize_log(self.run_id),
            sanitize_log(self.task_id),
            sanitize_log(self.agent_id),
        )


# SQLite schema for a single agent's run.db - see
# work/bernstein/sqlite-run-storage-design.md for the full multi-table
# design; this module implements the subset the instrumentation layer
# itself owns (llm_calls/tool_calls/messages/run_meta), matching the task
# spec's schema exactly.
_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_calls (
    call_id TEXT PRIMARY KEY,
    task_id TEXT,
    agent_id TEXT,
    ts_start TEXT,
    ts_end TEXT,
    wall_ms REAL,
    model TEXT,
    endpoint TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    status TEXT,
    error TEXT
);

CREATE TABLE IF NOT EXISTS tool_calls (
    call_id TEXT PRIMARY KEY,
    task_id TEXT,
    agent_id TEXT,
    ts_start TEXT,
    ts_end TEXT,
    wall_ms REAL,
    tool TEXT,
    args TEXT,
    result TEXT,
    success INTEGER,
    error TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    idx INTEGER,
    task_id TEXT,
    agent_id TEXT,
    role TEXT,
    content TEXT,
    content_length INTEGER,
    ts TEXT
);

CREATE TABLE IF NOT EXISTS run_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE INDEX IF NOT EXISTS idx_llm_calls_task   ON llm_calls(task_id);
CREATE INDEX IF NOT EXISTS idx_llm_calls_agent  ON llm_calls(agent_id);
CREATE INDEX IF NOT EXISTS idx_llm_calls_ts     ON llm_calls(ts_start);

CREATE INDEX IF NOT EXISTS idx_tool_calls_task  ON tool_calls(task_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_agent ON tool_calls(agent_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_ts    ON tool_calls(ts_start);

CREATE INDEX IF NOT EXISTS idx_messages_task    ON messages(task_id);
CREATE INDEX IF NOT EXISTS idx_messages_agent   ON messages(agent_id);
CREATE INDEX IF NOT EXISTS idx_messages_ts      ON messages(ts);
"""


class SQLiteInstrumenterBackend(InstrumenterBackend):
    """Writes instrumentation records into a single-file SQLite ``run.db``.

    One connection per target directory (``base_dir`` plus any
    ``extra_dirs`` for batched tasks - see module docstring), opened once at
    construction time and reused for every write; the instrumenter is
    already single-threaded per agent process (module docstring), so no
    connection pool or writer-thread/queue is needed here - a single
    ``threading.Lock`` still guards against two threads in the SAME process
    racing on the SAME connection (mirroring the JSONL backend's lock).

    Every write is wrapped in its own try/except: instrumentation must NEVER
    crash the calling agent, so a SQLite error (disk full, locked file,
    corrupt db) is logged with a full traceback and swallowed, exactly like
    the JSONL backend's OSError handling.

    DB location: ``<dir>/run.db`` for each target directory (the same
    ``base_dir``/``extra_dirs`` the caller already resolves via
    :func:`resolve_agent_dir` - today that is the per-agent directory, not
    the run root; see the class-level note in :func:`init_instrumenter` for
    why this stays consistent with the JSONL backend's existing directory
    convention instead of introducing a new one-db-per-run path resolution.)
    """

    def __init__(
        self,
        *,
        run_id: str,
        task_id: str,
        agent_id: str,
        base_dir: Path,
        extra_dirs: list[Path] | None = None,
    ) -> None:
        self.run_id = run_id
        self.task_id = task_id
        self.agent_id = agent_id
        self.base_dir = Path(base_dir)
        self.extra_dirs = [Path(d) for d in (extra_dirs or [])]
        self._lock = threading.Lock()
        self._connections: dict[Path, sqlite3.Connection] = {}

        all_dirs = [self.base_dir, *[d for d in self.extra_dirs if d != self.base_dir]]
        logger.debug(
            "SQLiteInstrumenterBackend: initializing run_id=%s task_id=%s agent_id=%s dirs=%s",
            sanitize_log(self.run_id),
            sanitize_log(self.task_id),
            sanitize_log(self.agent_id),
            all_dirs,
        )
        for target_dir in all_dirs:
            self._open_db(target_dir)

    def _open_db(self, target_dir: Path) -> None:
        db_path = target_dir / "run.db"
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA busy_timeout = 5000")
            conn.executescript(_SQLITE_SCHEMA)
            conn.commit()
            self._connections[target_dir] = conn
            logger.info(
                "SQLiteInstrumenterBackend: opened %s run_id=%s task_id=%s agent_id=%s (WAL mode, "
                "schema created/verified)",
                db_path,
                sanitize_log(self.run_id),
                sanitize_log(self.task_id),
                sanitize_log(self.agent_id),
            )
        except sqlite3.Error:
            logger.warning(
                "SQLiteInstrumenterBackend: failed to open/create %s (run_id=%s task_id=%s agent_id=%s) - "
                "instrumentation for this dir is DISABLED, the agent run itself is unaffected",
                db_path,
                sanitize_log(self.run_id),
                sanitize_log(self.task_id),
                sanitize_log(self.agent_id),
                exc_info=True,
            )
        except OSError:
            logger.warning(
                "SQLiteInstrumenterBackend: failed to create dir %s (run_id=%s task_id=%s agent_id=%s) - "
                "instrumentation for this dir is DISABLED, the agent run itself is unaffected",
                target_dir,
                sanitize_log(self.run_id),
                sanitize_log(self.task_id),
                sanitize_log(self.agent_id),
                exc_info=True,
            )

    def _execute_all(self, table: str, sql: str, params: tuple[Any, ...], *, key: str) -> None:
        if not self._connections:
            logger.debug(
                "SQLiteInstrumenterBackend: dropping %s write key=%s - no ready run.db connection "
                "(open failed or backend uninitialized)",
                table,
                sanitize_log(key),
            )
            return
        with self._lock:
            for target_dir, conn in self._connections.items():
                try:
                    conn.execute(sql, params)
                    conn.commit()
                    logger.debug(
                        "SQLiteInstrumenterBackend: wrote 1 row to %s (key=%s) in %s",
                        table,
                        sanitize_log(key),
                        target_dir / "run.db",
                    )
                except sqlite3.Error:
                    logger.warning(
                        "SQLiteInstrumenterBackend: failed to write %s row key=%s to %s",
                        table,
                        sanitize_log(key),
                        target_dir / "run.db",
                        exc_info=True,
                    )

    # -- InstrumenterBackend interface -----------------------------------

    def log_llm_call(
        self,
        *,
        call_id: str,
        ts_start: str,
        ts_end: str,
        wall_ms: float | None,
        model: str,
        endpoint: str | None,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
        status: str,
        error: str | None,
        ts_ttft: str | None = None,
        tokens_per_sec: float | None = None,
    ) -> None:
        # ts_ttft/tokens_per_sec have no column in the task-spec'd llm_calls
        # schema (streaming-only fields) - dropped here with a debug log
        # rather than silently lost; the JSONL backend still preserves them.
        if ts_ttft is not None or tokens_per_sec is not None:
            logger.debug(
                "SQLiteInstrumenterBackend.log_llm_call: dropping ts_ttft=%s tokens_per_sec=%s for call_id=%s "
                "(no column in llm_calls schema; JSONL backend preserves these)",
                ts_ttft,
                tokens_per_sec,
                sanitize_log(call_id),
            )
        self._execute_all(
            "llm_calls",
            "INSERT OR REPLACE INTO llm_calls "
            "(call_id, task_id, agent_id, ts_start, ts_end, wall_ms, model, endpoint, "
            "prompt_tokens, completion_tokens, total_tokens, status, error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                call_id,
                self.task_id,
                self.agent_id,
                ts_start,
                ts_end,
                wall_ms,
                model,
                endpoint,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                status,
                error,
            ),
            key=call_id,
        )

    def log_tool_call(
        self,
        *,
        call_id: str,
        ts_start: str,
        ts_end: str,
        wall_ms: float | None,
        tool: str,
        args: dict[str, Any],
        result: str | None,
        success: bool,
        error: str | None,
    ) -> None:
        try:
            args_json = json.dumps(args, ensure_ascii=False, default=str)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "SQLiteInstrumenterBackend.log_tool_call: failed to serialize args for %s: %s",
                sanitize_log(call_id),
                exc,
            )
            args_json = "{}"
        self._execute_all(
            "tool_calls",
            "INSERT OR REPLACE INTO tool_calls "
            "(call_id, task_id, agent_id, ts_start, ts_end, wall_ms, tool, args, result, success, error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                call_id,
                self.task_id,
                self.agent_id,
                ts_start,
                ts_end,
                wall_ms,
                tool,
                args_json,
                result,
                1 if success else 0,
                error,
            ),
            key=call_id,
        )

    def log_message(
        self,
        *,
        idx: int,
        role: str,
        content: str | None,
        content_length: int,
        ts: str,
        tool_calls: list[str] | None = None,
    ) -> None:
        # tool_calls has no column in the task-spec'd messages schema -
        # dropped here with a debug log; the JSONL backend still preserves it.
        if tool_calls:
            logger.debug(
                "SQLiteInstrumenterBackend.log_message: dropping tool_calls=%s for idx=%s "
                "(no column in messages schema; JSONL backend preserves these)",
                tool_calls,
                idx,
            )
        self._execute_all(
            "messages",
            "INSERT INTO messages (idx, task_id, agent_id, role, content, content_length, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (idx, self.task_id, self.agent_id, role, content, content_length, ts),
            key=str(idx),
        )

    def close(self) -> None:
        with self._lock:
            for target_dir, conn in self._connections.items():
                try:
                    conn.commit()
                    conn.close()
                    logger.debug("SQLiteInstrumenterBackend: closed %s", target_dir / "run.db")
                except sqlite3.Error:
                    logger.warning("SQLiteInstrumenterBackend: error closing %s", target_dir / "run.db", exc_info=True)
            self._connections.clear()


def _make_backend(
    backend: str,
    *,
    run_id: str,
    task_id: str,
    agent_id: str,
    base_dir: Path,
    extra_dirs: list[Path] | None,
) -> InstrumenterBackend:
    """Construct the requested backend, logging the selection decision."""
    if backend == "sqlite":
        logger.debug(
            "init_instrumenter: using sqlite backend run_id=%s task_id=%s agent_id=%s",
            sanitize_log(run_id),
            sanitize_log(task_id),
            sanitize_log(agent_id),
        )
        return SQLiteInstrumenterBackend(
            run_id=run_id, task_id=task_id, agent_id=agent_id, base_dir=base_dir, extra_dirs=extra_dirs
        )
    if backend == "jsonl":
        logger.debug(
            "init_instrumenter: using jsonl backend run_id=%s task_id=%s agent_id=%s",
            sanitize_log(run_id),
            sanitize_log(task_id),
            sanitize_log(agent_id),
        )
        return JSONLInstrumenterBackend(
            run_id=run_id, task_id=task_id, agent_id=agent_id, base_dir=base_dir, extra_dirs=extra_dirs
        )
    raise ValueError(f"init_instrumenter: unknown backend {backend!r} (expected 'sqlite' or 'jsonl')")


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------


class RunInstrumenter:
    """Facade over a pluggable :class:`InstrumenterBackend`.

    One instance is expected per (run_id, task_id, agent_id) triple - i.e.
    per agent process/session. Public method signatures are unchanged from
    the pre-refactor, JSONL-only implementation so every existing call site
    (:mod:`bernstein.adapters.openai_agents_runner`) keeps working without
    modification.

    Every public method is defensive: a failure anywhere in the backend is
    logged at WARNING and swallowed, never raised, so a full disk /
    permissions problem / serialization bug can never take down the agent
    run being observed.
    """

    def __init__(
        self,
        *,
        run_id: str,
        task_id: str,
        agent_id: str,
        base_dir: Path,
        extra_dirs: list[Path] | None = None,
        backend: str = _DEFAULT_BACKEND,
    ) -> None:
        self.run_id = run_id
        self.task_id = task_id
        self.agent_id = agent_id
        self.base_dir = Path(base_dir)
        self.extra_dirs = [Path(d) for d in (extra_dirs or [])]
        self.backend_name = backend
        self._backend: InstrumenterBackend | None = _make_backend(
            backend,
            run_id=run_id,
            task_id=task_id,
            agent_id=agent_id,
            base_dir=base_dir,
            extra_dirs=extra_dirs,
        )

    # -- public API -------------------------------------------------------

    def log_llm_call(
        self,
        *,
        call_id: str,
        ts_start: str,
        ts_end: str,
        model: str,
        endpoint: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        wall_ms: float | None = None,
        ts_ttft: str | None = None,
        tokens_per_sec: float | None = None,
        status: str = "ok",
        error: str | None = None,
    ) -> None:
        """Record one LLM API call.

        ``wall_ms`` is computed from ``ts_start``/``ts_end`` when not given
        explicitly. ``tokens_per_sec`` is only ever computed by the CALLER
        from real streamed first-token timestamps - this method never
        fabricates TTFT/throughput when the call did not stream (per task
        spec: "If it doesn't stream ... do NOT fake TTFT").
        """
        try:
            if wall_ms is None:
                wall_ms = _iso_delta_ms(ts_start, ts_end)
            if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
                total_tokens = prompt_tokens + completion_tokens
            if self._backend is None:
                logger.debug(
                    "RunInstrumenter.log_llm_call: no backend ready, dropping call_id=%s", sanitize_log(call_id)
                )
                return
            self._backend.log_llm_call(
                call_id=call_id,
                ts_start=ts_start,
                ts_end=ts_end,
                wall_ms=wall_ms,
                model=model,
                endpoint=endpoint,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                status=status,
                error=error,
                ts_ttft=ts_ttft,
                tokens_per_sec=tokens_per_sec,
            )
        except Exception as exc:  # intentional-broad-except: instrumentation must never raise
            logger.warning("RunInstrumenter.log_llm_call failed for call_id=%s: %s", sanitize_log(call_id), exc)

    def log_tool_call(
        self,
        *,
        call_id: str,
        ts_start: str,
        ts_end: str,
        tool: str,
        args: dict[str, Any] | None = None,
        success: bool,
        error: str | None = None,
        wall_ms: float | None = None,
        result: Any = None,
    ) -> None:
        """Record one tool invocation.

        ``args`` is truncated (see :func:`_truncate_value`) before being
        written - never the full tool call arguments unbounded, only a
        capped preview.

        ``result`` (bug fix, 2026-07-04: see :data:`_TOOL_RESULT_TRUNCATE_CHARS`)
        is the tool's own return value - whatever the caller's tool
        implementation produced (a string, dict, etc.) - stringified and
        truncated to :data:`_TOOL_RESULT_TRUNCATE_CHARS` characters via
        :func:`_truncate_text` before being written under the ``result``
        key. ``None`` (the default) omits the key entirely, matching every
        call site that has no result to report (e.g. a failed call where
        the error already carries the relevant text).
        """
        try:
            if wall_ms is None:
                wall_ms = _iso_delta_ms(ts_start, ts_end)
            truncated_args = _truncate_value(args or {})
            truncated_result = _truncate_text(
                result, max_chars=_TOOL_RESULT_TRUNCATE_CHARS, field_name="tool_call.result", key=call_id
            )
            if truncated_result is None:
                logger.debug(
                    "RunInstrumenter.log_tool_call: no result value provided for call_id=%s tool=%s",
                    sanitize_log(call_id),
                    tool,
                )
            if self._backend is None:
                logger.debug(
                    "RunInstrumenter.log_tool_call: no backend ready, dropping call_id=%s", sanitize_log(call_id)
                )
                return
            self._backend.log_tool_call(
                call_id=call_id,
                ts_start=ts_start,
                ts_end=ts_end,
                wall_ms=wall_ms,
                tool=tool,
                args=truncated_args,
                result=truncated_result,
                success=success,
                error=error,
            )
        except Exception as exc:  # intentional-broad-except: instrumentation must never raise
            logger.warning("RunInstrumenter.log_tool_call failed for call_id=%s: %s", sanitize_log(call_id), exc)

    def log_message(
        self,
        *,
        idx: int,
        role: str,
        content_length: int,
        tool_calls: list[str] | None = None,
        ts: str | None = None,
        content: Any = None,
    ) -> None:
        """Record one new conversation message.

        Shape metadata (``role``/``content_length``/``tool_calls``) is
        always recorded. ``content`` (bug fix, 2026-07-04: see
        :data:`_MESSAGE_CONTENT_TRUNCATE_CHARS`) is optional actual message
        text - stringified via :func:`_truncate_text` and truncated to
        :data:`_MESSAGE_CONTENT_TRUNCATE_CHARS` characters before being
        written under the ``content`` key. Callers that only have shape
        metadata (or that intentionally withhold content for privacy/size
        reasons) pass ``None`` (the default) and the key is omitted
        entirely, preserving the original shape-only behavior.
        """
        try:
            resolved_ts = ts or _now_iso()
            truncated_content = _truncate_text(
                content, max_chars=_MESSAGE_CONTENT_TRUNCATE_CHARS, field_name="message.content", key=str(idx)
            )
            if truncated_content is None:
                logger.debug("RunInstrumenter.log_message: no content value provided for idx=%s role=%s", idx, role)
            if self._backend is None:
                logger.debug("RunInstrumenter.log_message: no backend ready, dropping idx=%s", idx)
                return
            self._backend.log_message(
                idx=idx,
                role=role,
                content=truncated_content,
                content_length=content_length,
                ts=resolved_ts,
                tool_calls=tool_calls,
            )
        except Exception as exc:  # intentional-broad-except: instrumentation must never raise
            logger.warning("RunInstrumenter.log_message failed for idx=%s: %s", idx, exc)

    def close(self) -> None:
        """Flush/cleanup the active backend. Safe to call multiple times."""
        if self._backend is None:
            return
        try:
            self._backend.close()
        except Exception as exc:
            logger.warning("RunInstrumenter.close failed: %s", exc)
        finally:
            self._backend = None


class _NullInstrumenter(RunInstrumenter):
    """No-op stand-in returned by :func:`get_instrumenter` before init.

    Lets call sites do ``get_instrumenter().log_llm_call(...)`` unconditionally
    without an ``if instrumenter is not None`` guard at every call site, while
    guaranteeing zero disk I/O (and zero directory/DB creation) until a real
    instrumenter is explicitly initialized via :func:`init_instrumenter`.
    """

    def __init__(self) -> None:  # intentionally skips backend creation, no base __init__ to call
        self.run_id = "uninitialized"
        self.task_id = "uninitialized"
        self.agent_id = "uninitialized"
        self.base_dir = Path()
        self.extra_dirs = []
        self.backend_name = "none"
        self._backend = None


_instrumenter_lock = threading.Lock()
_instrumenter: RunInstrumenter | None = None
_null_instrumenter = _NullInstrumenter()

# Count of get_instrumenter() calls that returned the NullInstrumenter
# fallback, for debug visibility into "did init_instrumenter ever run in
# this process" (see log line in get_instrumenter below). Not thread-safe
# (best-effort counter, not a correctness-critical value).
_null_fallback_count = 0


def resolve_agent_dir(workdir: Path, run_id: str, task_id: str, agent_id: str) -> Path:
    """Return the per-agent instrumentation directory for the given ids.

    Mirrors the wave-2 run layout (``.sdd/runs/<run_id>/...``) documented in
    :mod:`bernstein.core.orchestration.run_report`. Used as ``base_dir`` by
    both backends - the JSONL backend writes its three ``*.jsonl`` files
    directly here; the SQLite backend writes a single ``run.db`` here.

    Each id is sanitized to a single directory-name component first (see
    :func:`_sanitize_path_component`): ``run_id`` comes from the
    ``BERNSTEIN_RUN_ID`` environment variable and ``task_id``/``agent_id``
    from the runner manifest, so a value carrying path separators or ``..``
    segments must not be able to point the instrumentation tree outside
    ``<workdir>/.sdd/runs/``.
    """
    return (
        workdir
        / ".sdd"
        / "runs"
        / _sanitize_path_component(run_id)
        / "tasks"
        / _sanitize_path_component(task_id)
        / "agents"
        / _sanitize_path_component(agent_id)
    )


def init_instrumenter(
    *,
    run_id: str,
    task_id: str,
    agent_id: str,
    base_dir: Path,
    extra_dirs: list[Path] | None = None,
    backend: str = _DEFAULT_BACKEND,
) -> RunInstrumenter:
    """Create and install the process-wide :class:`RunInstrumenter` singleton.

    Safe to call more than once (e.g. a test re-initializing between cases);
    each call replaces the previous singleton (the previous backend is
    closed first so file handles/DB connections don't leak across
    re-initializations). Never raises - construction failures are caught
    inside the backend's own constructor and degrade to a disabled-but-
    non-crashing instance.

    ``backend`` selects the storage sink: ``"sqlite"`` (default) writes a
    single ``run.db`` per target dir; ``"jsonl"`` writes the original
    three-file-per-agent format. See
    ``work/bernstein/sqlite-run-storage-design.md`` for the full design.

    ``extra_dirs`` (bug 3 fix - see :class:`JSONLInstrumenterBackend`
    docstring on ``extra_dirs``): when the spawner batches multiple tasks
    onto one agent process, pass the OTHER tasks' agent dirs here so every
    task in the batch gets a full copy of this agent's instrumentation
    instead of only ``task_id``'s. Works identically for both backends -
    the SQLite backend opens one ``run.db`` connection per target dir,
    mirroring the JSONL backend's one-file-set per target dir.
    """
    global _instrumenter
    resolved_extra_dirs = list(extra_dirs or [])
    logger.info(
        "init_instrumenter called: run_id=%s task_id=%s agent_id=%s base_dir=%s extra_dirs=%s backend=%s "
        "(%d extra dir(s) requested for batch fan-out)",
        sanitize_log(run_id),
        sanitize_log(task_id),
        sanitize_log(agent_id),
        base_dir,
        resolved_extra_dirs,
        backend,
        len(resolved_extra_dirs),
    )
    with _instrumenter_lock:
        previous = _instrumenter
    if previous is not None:
        try:
            previous.close()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("init_instrumenter: error closing previous instrumenter: %s", exc)

    instrumenter = RunInstrumenter(
        run_id=run_id,
        task_id=task_id,
        agent_id=agent_id,
        base_dir=base_dir,
        extra_dirs=resolved_extra_dirs,
        backend=backend,
    )
    with _instrumenter_lock:
        _instrumenter = instrumenter
    return instrumenter


def get_instrumenter() -> RunInstrumenter:
    """Return the process-wide instrumenter, or a silent no-op if uninitialized.

    Callers should prefer this over reaching into module state directly so
    an un-instrumented context (e.g. a unit test that imports a hooked
    module without calling :func:`init_instrumenter`) degrades to "nothing
    is written" instead of an ``AttributeError``.

    Logs at DEBUG the first few times this falls back to the
    :class:`_NullInstrumenter` in a process where a hook fired before
    :func:`init_instrumenter` ran (or it never ran at all) - a silent
    NullInstrumenter fallback is exactly the failure mode behind bug 3
    ("zero instrumentation" for some tasks), so every fallback is now
    visible in the logs rather than swallowed.
    """
    global _null_fallback_count
    with _instrumenter_lock:
        if _instrumenter is not None:
            return _instrumenter
        _null_fallback_count += 1
        if _null_fallback_count <= 10 or _null_fallback_count % 100 == 0:
            logger.debug(
                "get_instrumenter: returning NullInstrumenter fallback (call #%d in this process) - "
                "init_instrumenter has not been called yet, or was never called; any hook firing "
                "right now writes NOTHING to disk",
                _null_fallback_count,
            )
        return _null_instrumenter


def update_global_index(runs_dir: Path, run_id: str, meta_dict: dict[str, Any]) -> None:
    """Upsert one run-summary row into the shared ``<runs_dir>/index.db``.

    Called after a run completes (or periodically) so cross-run queries
    (``bernstein runs list``, ``runs compare``) never need to glob and open
    every per-run ``run.db`` - see
    ``work/bernstein/sqlite-run-storage-design.md`` §2. ``index.db`` is a
    derived cache: every column here is recomputable from the per-run
    ``run.db`` files, so a failure to write here is logged and swallowed,
    never raised - it must not affect the calling run's own completion.

    ``meta_dict`` keys map directly onto the ``runs`` table columns:
    ``dir_name``, ``issue``, ``repo``, ``workflow``, ``model``, ``git_sha``,
    ``started_at``, ``finished_at``, ``total_duration_s``, ``status``,
    ``task_count``, ``agent_count``. Missing keys are stored as ``NULL``.
    """
    db_path = Path(runs_dir) / "index.db"
    logger.debug("update_global_index: upserting run_id=%s into %s meta=%s", sanitize_log(run_id), db_path, meta_dict)
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA busy_timeout = 5000")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    dir_name TEXT,
                    issue TEXT,
                    repo TEXT,
                    workflow TEXT,
                    model TEXT,
                    git_sha TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    total_duration_s REAL,
                    status TEXT,
                    task_count INTEGER,
                    agent_count INTEGER
                );
                CREATE INDEX IF NOT EXISTS idx_runs_issue   ON runs(issue);
                CREATE INDEX IF NOT EXISTS idx_runs_status  ON runs(status);
                CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at);
                """
            )
            conn.execute(
                """
                INSERT INTO runs (
                    run_id, dir_name, issue, repo, workflow, model, git_sha,
                    started_at, finished_at, total_duration_s, status, task_count, agent_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    dir_name=excluded.dir_name,
                    issue=excluded.issue,
                    repo=excluded.repo,
                    workflow=excluded.workflow,
                    model=excluded.model,
                    git_sha=excluded.git_sha,
                    started_at=excluded.started_at,
                    finished_at=excluded.finished_at,
                    total_duration_s=excluded.total_duration_s,
                    status=excluded.status,
                    task_count=excluded.task_count,
                    agent_count=excluded.agent_count
                """,
                (
                    run_id,
                    meta_dict.get("dir_name"),
                    meta_dict.get("issue"),
                    meta_dict.get("repo"),
                    meta_dict.get("workflow"),
                    meta_dict.get("model"),
                    meta_dict.get("git_sha"),
                    meta_dict.get("started_at"),
                    meta_dict.get("finished_at"),
                    meta_dict.get("total_duration_s"),
                    meta_dict.get("status"),
                    meta_dict.get("task_count"),
                    meta_dict.get("agent_count"),
                ),
            )
            conn.commit()
            logger.info("update_global_index: upserted run_id=%s into %s", sanitize_log(run_id), db_path)
        finally:
            conn.close()
    except sqlite3.Error:
        logger.warning("update_global_index: failed to upsert run_id=%s into %s", sanitize_log(run_id), db_path, exc_info=True)
    except OSError:
        logger.warning("update_global_index: failed to create parent dir for %s", db_path, exc_info=True)
