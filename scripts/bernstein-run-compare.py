#!/usr/bin/env python3
"""bernstein-run-compare: side-by-side diff of two Bernstein run directories'
aggregate instrumentation (tokens, cost, duration, tool calls, errors).

Imports the aggregation logic from bernstein-run-report.py (same directory)
rather than duplicating it — see the design note in the docs for why.

Usage:
    python3 bernstein-run-compare.py <run_dir_a> <run_dir_b> [--format table|json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Import the sibling report script as a module. bernstein-run-report.py has a
# hyphen in its filename so it can't be `import`ed directly; load it via
# importlib from its known path (same directory as this script).
import importlib.util

_report_path = Path(__file__).resolve().parent / "bernstein-run-report.py"
_spec = importlib.util.spec_from_file_location("bernstein_run_report", _report_path)
_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_report)  # type: ignore[union-attr]


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def fmt_usd(v: float) -> str:
    return f"${v:,.4f}"


def delta_str(a, b, fmt="{:,.0f}", pct=True):
    d = b - a
    sign = "+" if d >= 0 else ""
    s = f"{sign}{fmt.format(d)}"
    if pct and a:
        s += f" ({sign}{d / a * 100:.1f}%)"
    elif pct and not a and d:
        s += " (n/a, base was 0)"
    return s


def print_table(name_a: str, res_a: dict, name_b: str, res_b: dict):
    ta, tb = res_a["totals"], res_b["totals"]

    print(f"Comparing:")
    print(f"  A = {name_a}")
    print(f"  B = {name_b}")
    print()

    rows = [
        ("run.db files", ta["db_count"], tb["db_count"], "{:,.0f}", False),
        ("LLM calls", ta["llm_call_count"], tb["llm_call_count"], "{:,.0f}", True),
        ("Tool calls", ta["tool_call_count"], tb["tool_call_count"], "{:,.0f}", True),
        ("Prompt tokens", ta["prompt_tokens"], tb["prompt_tokens"], "{:,.0f}", True),
        ("Completion tokens", ta["completion_tokens"], tb["completion_tokens"], "{:,.0f}", True),
        ("Total tokens", ta["total_tokens"], tb["total_tokens"], "{:,.0f}", True),
        ("Est. cost (USD)", ta["cost_usd"], tb["cost_usd"], "{:,.4f}", True),
        ("Error count", ta["error_count"], tb["error_count"], "{:,.0f}", True),
    ]

    dur_a = sum(p["duration_ms"] for p in ta["phases"].values())
    dur_b = sum(p["duration_ms"] for p in tb["phases"].values())
    rows.append(("Total duration (ms)", dur_a, dur_b, "{:,.0f}", True))

    header = f"{'Metric':<22}{'A':>18}{'B':>18}{'Delta (B-A)':>24}"
    print(header)
    print("-" * len(header))
    for label, a, b, fmt, pct in rows:
        print(f"{label:<22}{fmt.format(a):>18}{fmt.format(b):>18}{delta_str(a, b, fmt, pct):>24}")

    print()
    print(f"Models — A: {', '.join(sorted(ta['models'])) or '(none)'}")
    print(f"Models — B: {', '.join(sorted(tb['models'])) or '(none)'}")

    print()
    all_phases = sorted(set(ta["phases"]) | set(tb["phases"]))
    print(f"{'Phase':<15}{'A dur(ms)':>14}{'B dur(ms)':>14}{'A llm':>8}{'B llm':>8}")
    print("-" * 59)
    for phase in all_phases:
        pa = ta["phases"].get(phase, {"duration_ms": 0, "llm_calls": 0})
        pb = tb["phases"].get(phase, {"duration_ms": 0, "llm_calls": 0})
        print(
            f"{phase:<15}{pa['duration_ms']:>14,.0f}{pb['duration_ms']:>14,.0f}"
            f"{pa['llm_calls']:>8}{pb['llm_calls']:>8}"
        )

    if ta["load_errors"] or tb["load_errors"]:
        print()
        print("Load errors:")
        for le in ta["load_errors"]:
            print(f"  A: {le['db_path']}: {le['error']}")
        for le in tb["load_errors"]:
            print(f"  B: {le['db_path']}: {le['error']}")


def print_json(name_a, res_a, name_b, res_b):
    out = {
        "a": {"name": name_a, "totals": _report.to_jsonable(res_a["totals"])},
        "b": {"name": name_b, "totals": _report.to_jsonable(res_b["totals"])},
    }
    print(json.dumps(out, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(
        description="Compare aggregate Bernstein instrumentation between two run directories."
    )
    parser.add_argument("run_dir_a")
    parser.add_argument("run_dir_b")
    parser.add_argument("--format", choices=["table", "json"], default="table")
    args = parser.parse_args()

    dir_a, dir_b = Path(args.run_dir_a), Path(args.run_dir_b)
    for d in (dir_a, dir_b):
        if not d.is_dir():
            eprint(f"error: run directory does not exist: {d}")
            sys.exit(1)

    res_a = _report.aggregate_run(dir_a)
    res_b = _report.aggregate_run(dir_b)

    if args.format == "table":
        print_table(str(dir_a), res_a, str(dir_b), res_b)
    else:
        print_json(str(dir_a), res_a, str(dir_b), res_b)


if __name__ == "__main__":
    main()
