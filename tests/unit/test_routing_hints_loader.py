"""Tests for the ``routing_hints.json`` consumer (loader + model selector).

Complements ``test_success_rate_advisor.py`` which covers the producer side.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from bernstein.evolution.routing_hints import (
    RoleRoutingHints,
    RoutingHints,
    load_routing_hints,
    select_model_with_hints,
)


# ---------------------------------------------------------------------------
# load_routing_hints — file states
# ---------------------------------------------------------------------------


def test_load_missing_file_returns_empty(tmp_path: Path) -> None:
    hints = load_routing_hints(tmp_path / "does_not_exist.json")
    assert hints.is_empty
    assert hints.for_role("backend") == RoleRoutingHints()


def test_load_invalid_json_returns_empty(tmp_path: Path) -> None:
    p = tmp_path / "hints.json"
    p.write_text("{not-json", encoding="utf-8")
    assert load_routing_hints(p).is_empty


def test_load_non_dict_top_level_returns_empty(tmp_path: Path) -> None:
    p = tmp_path / "hints.json"
    p.write_text('["not a dict"]', encoding="utf-8")
    assert load_routing_hints(p).is_empty


def test_load_stale_file_returns_empty(tmp_path: Path) -> None:
    p = tmp_path / "hints.json"
    p.write_text(
        json.dumps(
            {
                "generated_at": time.time() - 3600,
                "overall_success_rate": 0.2,
                "routing_recommendations": [
                    {
                        "role": "backend",
                        "model": "opus",
                        "recommendation": "avoid",
                        "success_rate": 0.0,
                        "failure_count": 3,
                        "reason": "…",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert load_routing_hints(p, max_age_seconds=60).is_empty


# ---------------------------------------------------------------------------
# load_routing_hints — parsing
# ---------------------------------------------------------------------------


def _write_hints(path: Path, recs: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "generated_at": time.time(),
                "overall_success_rate": 0.14,
                "total_tasks_analyzed": 7,
                "role_success_rates": {"backend": 0.14},
                "routing_recommendations": recs,
            }
        ),
        encoding="utf-8",
    )


def test_load_groups_avoid_and_prefer_per_role(tmp_path: Path) -> None:
    p = tmp_path / "routing_hints.json"
    _write_hints(
        p,
        [
            {"role": "backend", "model": "opus", "recommendation": "avoid"},
            {"role": "backend", "model": "sonnet", "recommendation": "prefer"},
            {"role": "frontend", "model": "haiku", "recommendation": "avoid"},
            {"role": "backend", "model": "haiku", "recommendation": "avoid"},
        ],
    )
    hints = load_routing_hints(p)
    backend = hints.for_role("backend")
    assert backend.avoid == frozenset({"opus", "haiku"})
    assert backend.prefer == frozenset({"sonnet"})
    frontend = hints.for_role("frontend")
    assert frontend.avoid == frozenset({"haiku"})
    assert frontend.prefer == frozenset()
    assert hints.for_role("unknown") == RoleRoutingHints()


def test_load_skips_malformed_entries(tmp_path: Path) -> None:
    p = tmp_path / "hints.json"
    _write_hints(
        p,
        [
            "not a dict",  # type: ignore[list-item]
            {"role": "backend"},  # missing fields
            {"role": 42, "model": "opus", "recommendation": "avoid"},
            {"role": "backend", "model": "opus", "recommendation": "avoid"},
        ],
    )
    hints = load_routing_hints(p)
    assert hints.for_role("backend").avoid == frozenset({"opus"})


def test_load_no_recommendations_key_returns_empty_role_map(tmp_path: Path) -> None:
    p = tmp_path / "hints.json"
    p.write_text(
        json.dumps({"generated_at": time.time(), "overall_success_rate": 0.9}),
        encoding="utf-8",
    )
    hints = load_routing_hints(p)
    assert hints.by_role == {}
    assert hints.overall_success_rate == 0.9


# ---------------------------------------------------------------------------
# select_model_with_hints
# ---------------------------------------------------------------------------


def _hints_for(role: str, *, avoid: tuple[str, ...] = (), prefer: tuple[str, ...] = ()) -> RoutingHints:
    return RoutingHints(
        generated_at=time.time(),
        overall_success_rate=0.5,
        by_role={role: RoleRoutingHints(avoid=frozenset(avoid), prefer=frozenset(prefer))},
    )


def test_selector_returns_candidate_when_not_avoided() -> None:
    hints = _hints_for("backend", avoid=("opus",), prefer=("sonnet",))
    assert select_model_with_hints("backend", "sonnet", hints) == "sonnet"
    assert select_model_with_hints("backend", "haiku", hints) == "haiku"


def test_selector_prefers_prefer_over_avoided_candidate() -> None:
    hints = _hints_for("backend", avoid=("opus",), prefer=("sonnet",))
    assert select_model_with_hints("backend", "opus", hints) == "sonnet"


def test_selector_falls_back_to_alt_when_no_prefer() -> None:
    hints = _hints_for("backend", avoid=("opus",))
    assert (
        select_model_with_hints(
            "backend", "opus", hints, fallback_candidates=("haiku", "sonnet")
        )
        == "haiku"
    )


def test_selector_skips_avoided_alt() -> None:
    hints = _hints_for("backend", avoid=("opus", "haiku"))
    assert (
        select_model_with_hints(
            "backend", "opus", hints, fallback_candidates=("haiku", "sonnet")
        )
        == "sonnet"
    )


def test_selector_keeps_candidate_when_no_viable_fallback() -> None:
    hints = _hints_for("backend", avoid=("opus",))
    assert select_model_with_hints("backend", "opus", hints) == "opus"


def test_selector_ignores_hints_for_other_role() -> None:
    hints = _hints_for("frontend", avoid=("opus",), prefer=("sonnet",))
    assert select_model_with_hints("backend", "opus", hints) == "opus"


def test_selector_prefer_does_not_reroute_when_candidate_equals_prefer() -> None:
    hints = _hints_for("backend", avoid=("opus",), prefer=("opus",))
    # Candidate is in prefer AND avoid — degenerate config. avoid wins,
    # but the only "prefer" alternative equals the candidate itself,
    # so we must fall through instead of returning the same string.
    assert (
        select_model_with_hints(
            "backend", "opus", hints, fallback_candidates=("sonnet",)
        )
        == "sonnet"
    )


def test_selector_prefer_iteration_is_deterministic() -> None:
    # Multiple prefer entries must resolve to the same choice on every call
    # (sorted alphabetically); frozenset iteration order alone is not stable.
    hints = _hints_for(
        "backend", avoid=("opus",), prefer=("sonnet", "haiku", "mistral")
    )
    picks = {select_model_with_hints("backend", "opus", hints) for _ in range(50)}
    assert picks == {"haiku"}


def test_selector_skips_prefer_that_is_also_avoided() -> None:
    # Degenerate producer state: a model landed in both avoid and prefer.
    # avoid must win — the selector must not return an avoided model.
    hints = _hints_for(
        "backend", avoid=("opus", "haiku"), prefer=("haiku", "sonnet")
    )
    assert select_model_with_hints("backend", "opus", hints) == "sonnet"
