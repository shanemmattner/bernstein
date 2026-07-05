"""Unified reader for per-agent instrumentation, regardless of backend format.

:mod:`bernstein.core.instrumentation` writes per-agent instrumentation using
one of two pluggable backends (see that module's docstring):

* :class:`~bernstein.core.instrumentation.SQLiteInstrumenterBackend` - a
  single ``run.db`` per agent directory.
* :class:`~bernstein.core.instrumentation.JSONLInstrumenterBackend` - three
  files per agent directory (``llm-calls.jsonl``, ``tool-calls.jsonl``,
  ``conversation.jsonl``).

Consumers that just want "the llm calls/tool calls/messages for this agent
directory" (e.g. the Bernstein TUI, ``scripts/bernstein-run-report.py`` and
friends, ad-hoc analysis) should not need to know or care which backend
wrote the data. This module provides that abstraction:

    from bernstein.core.instrumentation_reader import (
        read_llm_calls, read_tool_calls, read_messages, detect_format,
    )

    calls = read_llm_calls(agent_dir)   # -> list[dict], SQLite column shape

Format detection (see :func:`detect_format`): look for ``run.db`` first
(SQLite is the current default backend - see
``bernstein.core.instrumentation._DEFAULT_BACKEND``); fall back to any
``*.jsonl`` file in the directory; otherwise report "none". This module is
observe-only and defensive to the same standard as the write-side
instrumentation module (see ``.claude/rules/lessons.md`` rule 2 and the
module docstring of ``instrumentation.py``): every read function returns an
empty list rather than raising, for ANY failure mode - missing directory,
missing file, corrupt SQLite file, malformed JSON line, missing table. It
must always be safe to call these functions speculatively (e.g. the TUI
polling an agent directory that hasn't been instrumented yet, or one whose
instrumentation is still being written).

Output shape: every dict returned by ``read_llm_calls``/``read_tool_calls``/
``read_messages`` uses the SQLite column names as keys (see
``bernstein.core.instrumentation._SQLITE_SCHEMA`` - the authoritative
source for these names), regardless of which backend actually produced the
data. For the JSONL backend, this means mapping the JSONL writer's actual
field names (see ``JSONLInstrumenterBackend.log_llm_call`` /
``log_tool_call`` / ``log_message``) onto this shape, filling in any key the
JSONL format doesn't carry (e.g. ``task_id``/``agent_id``, which the JSONL
records don't carry per-row - unlike the SQLite schema - because they're
implicit in the directory path) with ``None`` or a path-derived value.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

# Filenames written by JSONLInstrumenterBackend (instrumentation.py) - kept
# as constants here rather than imported, so this reader has zero import-time
# dependency on the write-side module (a reader must never fail to import
# just because the writer module changed unrelated internals).
_JSONL_LLM_CALLS_FILENAME = "llm-calls.jsonl"
_JSONL_TOOL_CALLS_FILENAME = "tool-calls.jsonl"
_JSONL_CONVERSATION_FILENAME = "conversation.jsonl"

_SQLITE_DB_FILENAME = "run.db"

# Unified output shape, matching bernstein.core.instrumentation._SQLITE_SCHEMA
# column names exactly. Every returned dict has ALL of these keys, regardless
# of backend or which fields that backend actually recorded for a given row.
_LLM_CALL_KEYS = (
    "call_id",
    "task_id",
    "agent_id",
    "ts_start",
    "ts_end",
    "wall_ms",
    "model",
    "endpoint",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "status",
    "error",
)
_TOOL_CALL_KEYS = (
    "call_id",
    "task_id",
    "agent_id",
    "ts_start",
    "ts_end",
    "wall_ms",
    "tool",
    "args",
    "result",
    "success",
    "error",
)
_MESSAGE_KEYS = (
    "id",
    "idx",
    "task_id",
    "agent_id",
    "role",
    "content",
    "content_length",
    "ts",
)


def _infer_task_agent_ids(agent_dir: Path) -> tuple[str | None, str | None]:
    """Best-effort task_id/agent_id inference from the directory path.

    Mirrors the convention used by scripts/bernstein-run-report.py's
    infer_phase / db_path.parent.parent.parent.name path-derivation: the
    expected layout is
    ``.sdd/runs/<run_id>/tasks/<task_id>/agents/<agent_id>/``. JSONL records
    don't carry task_id/agent_id per-row (unlike the SQLite schema), so for
    the JSONL fallback path these are recovered from the directory
    structure. Never raises - returns (None, None) on any layout mismatch.
    """
    try:
        agent_id = agent_dir.name
        # Expected layout: .../tasks/<task_id>/agents/<agent_id>/
        if agent_dir.parent.name == "agents":
            task_id = agent_dir.parent.parent.name
        else:
            task_id = None
        logger.debug(
            "instrumentation_reader: inferred task_id=%s agent_id=%s from agent_dir=%s",
            task_id,
            agent_id,
            agent_dir,
        )
        return task_id, agent_id
    except Exception:  # pragma: no cover - defensive, Path attribute access essentially never raises
        logger.warning(
            "instrumentation_reader: failed to infer task_id/agent_id from agent_dir=%s",
            agent_dir,
            exc_info=True,
        )
        return None, None


def detect_format(agent_dir: Path) -> str:
    """Return "sqlite" | "jsonl" | "none" for the given agent directory.

    Detection order (see module docstring): run.db first (current default
    backend), then any *.jsonl file, then "none". Never raises.
    """
    agent_dir = Path(agent_dir)
    logger.debug("instrumentation_reader.detect_format: checking agent_dir=%s", agent_dir)
    try:
        if not agent_dir.is_dir():
            logger.debug(
                "instrumentation_reader.detect_format: agent_dir=%s is not a directory -> 'none'",
                agent_dir,
            )
            return "none"

        db_path = agent_dir / _SQLITE_DB_FILENAME
        if db_path.is_file():
            logger.info(
                "instrumentation_reader.detect_format: found %s -> 'sqlite' (agent_dir=%s)",
                db_path,
                agent_dir,
            )
            return "sqlite"

        jsonl_candidates = [
            agent_dir / _JSONL_LLM_CALLS_FILENAME,
            agent_dir / _JSONL_TOOL_CALLS_FILENAME,
            agent_dir / _JSONL_CONVERSATION_FILENAME,
        ]
        if any(p.is_file() for p in jsonl_candidates):
            logger.info(
                "instrumentation_reader.detect_format: found jsonl file(s) -> 'jsonl' (agent_dir=%s, present=%s)",
                agent_dir,
                [p.name for p in jsonl_candidates if p.is_file()],
            )
            return "jsonl"

        logger.info(
            "instrumentation_reader.detect_format: no run.db or *.jsonl found -> 'none' (agent_dir=%s)",
            agent_dir,
        )
        return "none"
    except Exception:  # pragma: no cover - defensive, filesystem checks essentially never raise beyond OSError
        logger.warning(
            "instrumentation_reader.detect_format: unexpected error probing agent_dir=%s, returning 'none'",
            agent_dir,
            exc_info=True,
        )
        return "none"


# ---------------------------------------------------------------------------
# SQLite read path
# ---------------------------------------------------------------------------


def _sqlite_table_exists(conn: sqlite3.Connection, table: str) -> bool:
    try:
        cur = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,))
        return cur.fetchone() is not None
    except sqlite3.Error:
        logger.warning(
            "instrumentation_reader: sqlite_master existence check failed for table=%s",
            table,
            exc_info=True,
        )
        return False


def _read_sqlite_rows(agent_dir: Path, table: str, columns: tuple[str, ...]) -> list[dict]:
    """Generic defensive SQLite row reader.

    Returns list of dicts keyed exactly by `columns`, in the same order as
    the SQLite row. Never raises: any sqlite3.Error, missing file, or
    missing table degrades to an empty list plus a logged warning.
    """
    db_path = agent_dir / _SQLITE_DB_FILENAME
    logger.debug(
        "instrumentation_reader._read_sqlite_rows: table=%s db_path=%s columns=%s",
        table,
        db_path,
        columns,
    )
    if not db_path.is_file():
        logger.warning(
            "instrumentation_reader._read_sqlite_rows: %s does not exist, returning [] for table=%s",
            db_path,
            table,
        )
        return []

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        logger.warning(
            "instrumentation_reader._read_sqlite_rows: failed to open %s (mode=ro), returning [] for table=%s",
            db_path,
            table,
            exc_info=True,
        )
        return []

    rows_out: list[dict] = []
    try:
        if not _sqlite_table_exists(conn, table):
            logger.warning(
                "instrumentation_reader._read_sqlite_rows: table=%s missing in %s, returning []",
                table,
                db_path,
            )
            return []
        col_list = ", ".join(columns)
        try:
            cur = conn.execute(f"SELECT {col_list} FROM {table}")
            for row in cur.fetchall():
                rows_out.append(dict(zip(columns, row)))
        except sqlite3.Error:
            logger.warning(
                "instrumentation_reader._read_sqlite_rows: query failed for table=%s in %s, "
                "returning whatever rows were read so far (%d)",
                table,
                db_path,
                len(rows_out),
                exc_info=True,
            )
    finally:
        conn.close()

    logger.info(
        "instrumentation_reader._read_sqlite_rows: read %d row(s) from table=%s in %s",
        len(rows_out),
        table,
        db_path,
    )
    return rows_out


def _read_sqlite_llm_calls(agent_dir: Path) -> list[dict]:
    return _read_sqlite_rows(agent_dir, "llm_calls", _LLM_CALL_KEYS)


def _read_sqlite_tool_calls(agent_dir: Path) -> list[dict]:
    return _read_sqlite_rows(agent_dir, "tool_calls", _TOOL_CALL_KEYS)


def _read_sqlite_messages(agent_dir: Path) -> list[dict]:
    return _read_sqlite_rows(agent_dir, "messages", _MESSAGE_KEYS)


# ---------------------------------------------------------------------------
# JSONL read path
# ---------------------------------------------------------------------------


def _read_jsonl_lines(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts. Never raises.

    A single malformed line is logged and skipped (not fatal for the whole
    file) - matches the writer's own "never let one bad record break
    everything" defensive posture.
    """
    if not path.is_file():
        logger.debug(
            "instrumentation_reader._read_jsonl_lines: %s does not exist, returning []",
            path,
        )
        return []

    records: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except (json.JSONDecodeError, ValueError):
                    logger.warning(
                        "instrumentation_reader._read_jsonl_lines: failed to parse line %d of %s, skipping that line",
                        lineno,
                        path,
                        exc_info=True,
                    )
    except OSError:
        logger.warning(
            "instrumentation_reader._read_jsonl_lines: failed to open/read %s, returning %d record(s) parsed so far",
            path,
            len(records),
            exc_info=True,
        )

    logger.info(
        "instrumentation_reader._read_jsonl_lines: parsed %d record(s) from %s",
        len(records),
        path,
    )
    return records


def _read_jsonl_llm_calls(agent_dir: Path) -> list[dict]:
    """Map JSONLInstrumenterBackend.log_llm_call records onto the unified shape.

    Source fields (see instrumentation.py JSONLInstrumenterBackend.log_llm_call):
    call_id, ts_start, ts_end, wall_ms, model, endpoint, prompt_tokens,
    completion_tokens, total_tokens, status, error (+ optional ts_ttft,
    tokens_per_sec, which have no column in the unified/SQLite shape and are
    dropped here, mirroring what SQLiteInstrumenterBackend.log_llm_call
    itself does).
    """
    task_id, agent_id = _infer_task_agent_ids(agent_dir)
    raw = _read_jsonl_lines(agent_dir / _JSONL_LLM_CALLS_FILENAME)
    out = []
    for rec in raw:
        out.append(
            {
                "call_id": rec.get("call_id"),
                "task_id": task_id,
                "agent_id": agent_id,
                "ts_start": rec.get("ts_start"),
                "ts_end": rec.get("ts_end"),
                "wall_ms": rec.get("wall_ms"),
                "model": rec.get("model"),
                "endpoint": rec.get("endpoint"),
                "prompt_tokens": rec.get("prompt_tokens"),
                "completion_tokens": rec.get("completion_tokens"),
                "total_tokens": rec.get("total_tokens"),
                "status": rec.get("status"),
                "error": rec.get("error"),
            }
        )
    logger.debug(
        "instrumentation_reader._read_jsonl_llm_calls: mapped %d record(s) for agent_dir=%s",
        len(out),
        agent_dir,
    )
    return out


def _read_jsonl_tool_calls(agent_dir: Path) -> list[dict]:
    """Map JSONLInstrumenterBackend.log_tool_call records onto the unified shape.

    Source fields: call_id, ts_start, ts_end, wall_ms, tool, args, success,
    error (+ optional result, only present when not None in the original
    write - see log_tool_call).
    """
    task_id, agent_id = _infer_task_agent_ids(agent_dir)
    raw = _read_jsonl_lines(agent_dir / _JSONL_TOOL_CALLS_FILENAME)
    out = []
    for rec in raw:
        # args is stored as a JSON-serializable structure (dict/list), not a
        # string, in the JSONL writer - unlike the SQLite backend, which
        # stores args as a JSON-encoded TEXT column. Serialize here so the
        # unified shape is consistent (args as a JSON string) regardless of
        # backend.
        args_val = rec.get("args")
        if args_val is not None and not isinstance(args_val, str):
            try:
                args_val = json.dumps(args_val, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                logger.warning(
                    "instrumentation_reader._read_jsonl_tool_calls: failed to re-serialize args "
                    "for call_id=%s, leaving as None",
                    rec.get("call_id"),
                    exc_info=True,
                )
                args_val = None
        out.append(
            {
                "call_id": rec.get("call_id"),
                "task_id": task_id,
                "agent_id": agent_id,
                "ts_start": rec.get("ts_start"),
                "ts_end": rec.get("ts_end"),
                "wall_ms": rec.get("wall_ms"),
                "tool": rec.get("tool"),
                "args": args_val,
                "result": rec.get("result"),
                "success": rec.get("success"),
                "error": rec.get("error"),
            }
        )
    logger.debug(
        "instrumentation_reader._read_jsonl_tool_calls: mapped %d record(s) for agent_dir=%s",
        len(out),
        agent_dir,
    )
    return out


def _read_jsonl_messages(agent_dir: Path) -> list[dict]:
    """Map JSONLInstrumenterBackend.log_message records onto the unified shape.

    Source fields: idx, role, content_length, ts (+ optional content,
    tool_calls). There is no autoincrement `id` in the JSONL format (that's
    a SQLite-only surrogate key) - filled with None. `tool_calls` (a list of
    call_ids referenced by an assistant message) has no column in the
    unified/SQLite messages shape either and is dropped here for the same
    reason ts_ttft/tokens_per_sec are dropped for llm_calls.
    """
    task_id, agent_id = _infer_task_agent_ids(agent_dir)
    raw = _read_jsonl_lines(agent_dir / _JSONL_CONVERSATION_FILENAME)
    out = []
    for rec in raw:
        out.append(
            {
                "id": None,
                "idx": rec.get("idx"),
                "task_id": task_id,
                "agent_id": agent_id,
                "role": rec.get("role"),
                "content": rec.get("content"),
                "content_length": rec.get("content_length"),
                "ts": rec.get("ts"),
            }
        )
    logger.debug(
        "instrumentation_reader._read_jsonl_messages: mapped %d record(s) for agent_dir=%s",
        len(out),
        agent_dir,
    )
    return out


# ---------------------------------------------------------------------------
# Public API - format-agnostic dispatch
# ---------------------------------------------------------------------------


def read_llm_calls(agent_dir: Path) -> list[dict]:
    """Return all LLM-call records for agent_dir, regardless of backend format.

    Auto-detects format via :func:`detect_format`. Returns [] (never raises)
    when the format is "none" or any read fails.
    """
    agent_dir = Path(agent_dir)
    fmt = detect_format(agent_dir)
    logger.info(
        "instrumentation_reader.read_llm_calls: agent_dir=%s detected format=%s",
        agent_dir,
        fmt,
    )
    if fmt == "sqlite":
        rows = _read_sqlite_llm_calls(agent_dir)
    elif fmt == "jsonl":
        rows = _read_jsonl_llm_calls(agent_dir)
    else:
        rows = []
    logger.info(
        "instrumentation_reader.read_llm_calls: agent_dir=%s format=%s -> %d row(s)",
        agent_dir,
        fmt,
        len(rows),
    )
    return rows


def read_tool_calls(agent_dir: Path) -> list[dict]:
    """Return all tool-call records for agent_dir, regardless of backend format.

    Auto-detects format via :func:`detect_format`. Returns [] (never raises)
    when the format is "none" or any read fails.
    """
    agent_dir = Path(agent_dir)
    fmt = detect_format(agent_dir)
    logger.info(
        "instrumentation_reader.read_tool_calls: agent_dir=%s detected format=%s",
        agent_dir,
        fmt,
    )
    if fmt == "sqlite":
        rows = _read_sqlite_tool_calls(agent_dir)
    elif fmt == "jsonl":
        rows = _read_jsonl_tool_calls(agent_dir)
    else:
        rows = []
    logger.info(
        "instrumentation_reader.read_tool_calls: agent_dir=%s format=%s -> %d row(s)",
        agent_dir,
        fmt,
        len(rows),
    )
    return rows


def read_messages(agent_dir: Path) -> list[dict]:
    """Return all conversation-message records for agent_dir, regardless of
    backend format.

    Auto-detects format via :func:`detect_format`. Returns [] (never raises)
    when the format is "none" or any read fails.
    """
    agent_dir = Path(agent_dir)
    fmt = detect_format(agent_dir)
    logger.info(
        "instrumentation_reader.read_messages: agent_dir=%s detected format=%s",
        agent_dir,
        fmt,
    )
    if fmt == "sqlite":
        rows = _read_sqlite_messages(agent_dir)
    elif fmt == "jsonl":
        rows = _read_jsonl_messages(agent_dir)
    else:
        rows = []
    logger.info(
        "instrumentation_reader.read_messages: agent_dir=%s format=%s -> %d row(s)",
        agent_dir,
        fmt,
        len(rows),
    )
    return rows


__all__ = [
    "detect_format",
    "read_llm_calls",
    "read_messages",
    "read_tool_calls",
]
