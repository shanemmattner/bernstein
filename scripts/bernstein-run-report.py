#!/usr/bin/env python3
"""bernstein-run-report: aggregate per-agent SQLite instrumentation for a
single Bernstein run into a human/machine-readable summary.

Reads .sdd/runs/<run-id>/tasks/*/agents/*/run.db files. Stdlib + sqlite3
only, no third-party deps.

Usage:
    python3 bernstein-run-report.py [run_dir] [--format table|json|csv] [--detail]

If run_dir is omitted, auto-discovers the latest run directory under
./.sdd/runs/ (cwd-relative), sorting by run-id string (timestamp-like,
e.g. 20260704-195112, which sorts correctly lexicographically).
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Pricing table — $ per 1M tokens (prompt, completion). NEEDS PERIODIC MANUAL
# UPDATES. Unmapped models report "unknown model, cost N/A" rather than
# silently guessing.
# ---------------------------------------------------------------------------
PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "claude-sonnet-4": (3.00, 15.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-opus-4": (15.00, 75.00),
    "claude-opus-4-1": (15.00, 75.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "o1": (15.00, 60.00),
    "MiniMax-M2.7-highspeed": (0.30, 1.20),
    "MiniMax-M2.7": (0.30, 1.20),
    "deepseek-v4": (0.28, 0.42),
}

KNOWN_TABLES = {"llm_calls", "tool_calls", "messages", "run_meta"}


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def find_latest_run_dir(base: Path = Path(".sdd/runs")) -> Path | None:
    """Auto-discover the latest run dir under base, sorted by run-id string
    (timestamp-like names such as 20260704-195112 sort correctly as text)."""
    if not base.is_dir():
        return None
    candidates = [p for p in base.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name)
    return candidates[-1]


def discover_run_dbs(run_dir: Path) -> list[Path]:
    """Find all tasks/*/agents/*/run.db files under run_dir."""
    return sorted(run_dir.glob("tasks/*/agents/*/run.db"))


def infer_phase(db_path: Path) -> str:
    """Infer the pipeline phase (triage/implement/review/...) from the
    agent directory name, e.g. 'implement-252fe582' -> 'implement'."""
    agent_dir_name = db_path.parent.name  # e.g. "implement-252fe582"
    if "-" in agent_dir_name:
        return agent_dir_name.rsplit("-", 1)[0]
    return agent_dir_name


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    try:
        cur = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        )
        return cur.fetchone() is not None
    except sqlite3.Error as exc:
        eprint(f"  warning: sqlite_master check failed: {exc}")
        return False


def parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def cost_for_row(model: str | None, prompt_tokens: int, completion_tokens: int):
    """Returns (cost_usd_or_None, note)."""
    if not model:
        return None, "no model recorded"
    if model not in PRICING_PER_1M:
        return None, f"unknown model '{model}', cost N/A"
    in_price, out_price = PRICING_PER_1M[model]
    cost = (prompt_tokens or 0) / 1_000_000 * in_price + (
        completion_tokens or 0
    ) / 1_000_000 * out_price
    return cost, None


def aggregate_db(db_path: Path) -> dict:
    """Aggregate a single run.db into a summary dict. Never raises — all
    sqlite errors are caught and surfaced as an 'errors' list entry."""
    summary = {
        "db_path": str(db_path),
        "phase": infer_phase(db_path),
        "agent_dir": db_path.parent.name,
        "task_id": db_path.parent.parent.parent.name,
        "llm_call_count": 0,
        "tool_call_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "cost_unknown_models": set(),
        "models": set(),
        "wall_ms_start": None,
        "wall_ms_end": None,
        "duration_ms": None,
        "errors": [],
        "missing_tables": [],
        "load_error": None,
    }

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        summary["load_error"] = f"failed to open db: {exc}"
        return summary

    try:
        for tbl in KNOWN_TABLES:
            if not table_exists(conn, tbl):
                summary["missing_tables"].append(tbl)

        if "llm_calls" not in summary["missing_tables"]:
            try:
                cur = conn.execute(
                    "SELECT model, prompt_tokens, completion_tokens, total_tokens, "
                    "status, error, ts_start, ts_end FROM llm_calls"
                )
                for row in cur.fetchall():
                    model, pt, ct, tt, status, error, ts_start, ts_end = row
                    summary["llm_call_count"] += 1
                    pt, ct, tt = pt or 0, ct or 0, tt or 0
                    summary["prompt_tokens"] += pt
                    summary["completion_tokens"] += ct
                    summary["total_tokens"] += tt
                    if model:
                        summary["models"].add(model)
                    cost, note = cost_for_row(model, pt, ct)
                    if cost is not None:
                        summary["cost_usd"] += cost
                    elif model:
                        summary["cost_unknown_models"].add(model)
                    if error:
                        summary["errors"].append(
                            {"kind": "llm_call", "detail": error}
                        )
                    if status and status not in ("ok", "success"):
                        summary["errors"].append(
                            {"kind": "llm_call_status", "detail": f"status={status}"}
                        )
                    st = parse_ts(ts_start)
                    en = parse_ts(ts_end)
                    if st and (
                        summary["wall_ms_start"] is None or st < summary["wall_ms_start"]
                    ):
                        summary["wall_ms_start"] = st
                    if en and (
                        summary["wall_ms_end"] is None or en > summary["wall_ms_end"]
                    ):
                        summary["wall_ms_end"] = en
            except sqlite3.Error as exc:
                summary["errors"].append({"kind": "query_error", "detail": f"llm_calls: {exc}"})

        if "tool_calls" not in summary["missing_tables"]:
            try:
                cur = conn.execute(
                    "SELECT tool, success, error, ts_start, ts_end FROM tool_calls"
                )
                for tool, success, error, ts_start, ts_end in cur.fetchall():
                    summary["tool_call_count"] += 1
                    if (success is not None and success == 0) or error:
                        summary["errors"].append(
                            {"kind": "tool_call", "detail": f"tool={tool} error={error}"}
                        )
                    st = parse_ts(ts_start)
                    en = parse_ts(ts_end)
                    if st and (
                        summary["wall_ms_start"] is None or st < summary["wall_ms_start"]
                    ):
                        summary["wall_ms_start"] = st
                    if en and (
                        summary["wall_ms_end"] is None or en > summary["wall_ms_end"]
                    ):
                        summary["wall_ms_end"] = en
            except sqlite3.Error as exc:
                summary["errors"].append({"kind": "query_error", "detail": f"tool_calls: {exc}"})

    finally:
        conn.close()

    if summary["wall_ms_start"] and summary["wall_ms_end"]:
        summary["duration_ms"] = (
            summary["wall_ms_end"] - summary["wall_ms_start"]
        ).total_seconds() * 1000

    return summary


def aggregate_run(run_dir: Path) -> dict:
    """Aggregate every run.db under run_dir into one run-level summary plus
    a per-agent breakdown list."""
    dbs = discover_run_dbs(run_dir)
    per_agent = [aggregate_db(db) for db in dbs]

    totals = {
        "run_dir": str(run_dir),
        "db_count": len(dbs),
        "llm_call_count": 0,
        "tool_call_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "cost_unknown_models": set(),
        "models": set(),
        "phases": {},  # phase -> {duration_ms, llm_calls, tool_calls}
        "error_count": 0,
        "errors": [],
        "load_errors": [],
    }

    for agent in per_agent:
        if agent["load_error"]:
            totals["load_errors"].append(
                {"db_path": agent["db_path"], "error": agent["load_error"]}
            )
            continue
        totals["llm_call_count"] += agent["llm_call_count"]
        totals["tool_call_count"] += agent["tool_call_count"]
        totals["prompt_tokens"] += agent["prompt_tokens"]
        totals["completion_tokens"] += agent["completion_tokens"]
        totals["total_tokens"] += agent["total_tokens"]
        totals["cost_usd"] += agent["cost_usd"]
        totals["cost_unknown_models"] |= agent["cost_unknown_models"]
        totals["models"] |= agent["models"]
        totals["error_count"] += len(agent["errors"])
        for err in agent["errors"]:
            totals["errors"].append({**err, "agent_dir": agent["agent_dir"], "task_id": agent["task_id"]})

        phase = agent["phase"]
        pdata = totals["phases"].setdefault(
            phase, {"duration_ms": 0.0, "llm_calls": 0, "tool_calls": 0, "agent_count": 0}
        )
        pdata["duration_ms"] += agent["duration_ms"] or 0.0
        pdata["llm_calls"] += agent["llm_call_count"]
        pdata["tool_calls"] += agent["tool_call_count"]
        pdata["agent_count"] += 1

    return {"totals": totals, "per_agent": per_agent}


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def fmt_usd(v: float) -> str:
    return f"${v:,.4f}"


def print_table(result: dict, detail: bool):
    t = result["totals"]
    print(f"Run directory : {t['run_dir']}")
    print(f"run.db files  : {t['db_count']}")
    if t["load_errors"]:
        print(f"  ! {len(t['load_errors'])} db file(s) failed to load:")
        for le in t["load_errors"]:
            print(f"    - {le['db_path']}: {le['error']}")
    print()
    print(f"{'Metric':<28}{'Value':>20}")
    print("-" * 48)
    print(f"{'LLM calls':<28}{t['llm_call_count']:>20,}")
    print(f"{'Tool calls':<28}{t['tool_call_count']:>20,}")
    print(f"{'Prompt tokens':<28}{t['prompt_tokens']:>20,}")
    print(f"{'Completion tokens':<28}{t['completion_tokens']:>20,}")
    print(f"{'Total tokens':<28}{t['total_tokens']:>20,}")
    print(f"{'Estimated cost':<28}{fmt_usd(t['cost_usd']):>20}")
    print(f"{'Errors':<28}{t['error_count']:>20,}")
    print()
    print(f"Models used     : {', '.join(sorted(t['models'])) or '(none recorded)'}")
    if t["cost_unknown_models"]:
        print(f"Cost N/A for    : {', '.join(sorted(t['cost_unknown_models']))} (add to PRICING_PER_1M)")
    print()
    print(f"{'Phase':<15}{'Agents':>8}{'LLM calls':>12}{'Tool calls':>12}{'Duration(ms)':>15}")
    print("-" * 62)
    for phase, pdata in sorted(t["phases"].items()):
        print(
            f"{phase:<15}{pdata['agent_count']:>8}{pdata['llm_calls']:>12,}"
            f"{pdata['tool_calls']:>12,}{pdata['duration_ms']:>15,.0f}"
        )

    if t["errors"]:
        print()
        print(f"Errors ({len(t['errors'])}):")
        for e in t["errors"][:20]:
            print(f"  - [{e['kind']}] task={e['task_id']} agent={e['agent_dir']}: {e['detail'][:120]}")
        if len(t["errors"]) > 20:
            print(f"  ... and {len(t['errors']) - 20} more")

    if detail:
        print()
        print("Per-agent breakdown:")
        header = f"{'task_id':<14}{'agent_dir':<24}{'phase':<12}{'llm':>6}{'tool':>6}{'tokens':>10}{'cost':>12}{'dur(ms)':>12}{'errs':>6}"
        print(header)
        print("-" * len(header))
        for agent in result["per_agent"]:
            if agent["load_error"]:
                print(f"{agent['task_id']:<14}{agent['agent_dir']:<24} LOAD ERROR: {agent['load_error']}")
                continue
            print(
                f"{agent['task_id']:<14}{agent['agent_dir']:<24}{agent['phase']:<12}"
                f"{agent['llm_call_count']:>6}{agent['tool_call_count']:>6}"
                f"{agent['total_tokens']:>10,}{fmt_usd(agent['cost_usd']):>12}"
                f"{(agent['duration_ms'] or 0):>12,.0f}{len(agent['errors']):>6}"
            )
            if agent["missing_tables"]:
                print(f"    ! missing tables: {', '.join(agent['missing_tables'])}")


def to_jsonable(result: dict) -> dict:
    """Convert sets to sorted lists for JSON serialization."""
    def conv(obj):
        if isinstance(obj, set):
            return sorted(obj)
        if isinstance(obj, dict):
            return {k: conv(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [conv(v) for v in obj]
        return obj

    return conv(result)


def print_json(result: dict):
    print(json.dumps(to_jsonable(result), indent=2, default=str))


def print_csv(result: dict):
    writer = csv.writer(sys.stdout)
    writer.writerow(["task_id", "agent_dir", "phase", "llm_calls", "tool_calls",
                      "prompt_tokens", "completion_tokens", "total_tokens",
                      "cost_usd", "duration_ms", "error_count", "models"])
    for agent in result["per_agent"]:
        if agent["load_error"]:
            writer.writerow([agent["task_id"], agent["agent_dir"], "LOAD_ERROR",
                              "", "", "", "", "", "", "", "", agent["load_error"]])
            continue
        writer.writerow([
            agent["task_id"], agent["agent_dir"], agent["phase"],
            agent["llm_call_count"], agent["tool_call_count"],
            agent["prompt_tokens"], agent["completion_tokens"], agent["total_tokens"],
            f"{agent['cost_usd']:.6f}", agent["duration_ms"] or "",
            len(agent["errors"]), "|".join(sorted(agent["models"])),
        ])


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate Bernstein per-agent SQLite instrumentation for one run."
    )
    parser.add_argument("run_dir", nargs="?", default=None,
                         help="Path to a run directory (.sdd/runs/<run-id>). "
                              "If omitted, auto-discovers the latest under ./.sdd/runs/")
    parser.add_argument("--format", choices=["table", "json", "csv"], default="table")
    parser.add_argument("--detail", action="store_true",
                         help="Include per-agent breakdown table")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = find_latest_run_dir()
        if run_dir is None:
            eprint("error: no run_dir given and no runs found under ./.sdd/runs/")
            sys.exit(1)
        eprint(f"(auto-discovered latest run: {run_dir})")

    if not run_dir.is_dir():
        eprint(f"error: run directory does not exist: {run_dir}")
        sys.exit(1)

    result = aggregate_run(run_dir)

    if result["totals"]["db_count"] == 0:
        eprint(f"warning: no run.db files found under {run_dir}/tasks/*/agents/*/run.db")

    if args.format == "table":
        print_table(result, args.detail)
    elif args.format == "json":
        print_json(result)
    elif args.format == "csv":
        print_csv(result)


if __name__ == "__main__":
    main()
