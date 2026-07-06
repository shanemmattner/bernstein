"""Tests for spawner_core's SuccessRateAdvisor routing-hints consumer.

Wiring covered here (producer side lives in
``bernstein.evolution.detector.SuccessRateAdvisor``; consumer helper lives
in ``bernstein.evolution.routing_hints``):

* When ``<workdir>/.sdd/evolution/routing_hints.json`` marks
  ``(role="backend", model="opus")`` as "avoid", the spawner's model
  selection swaps opus out for another Claude tier.
* When the hints file is missing, stale, or empty, the current model is
  returned unchanged.
* Non-Claude candidates fall back to the ``role_model_policy["default"]``
  model, keeping the swap inside the adapter's family.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from bernstein.core.agents.spawner_core import _apply_routing_hints


def _write_hints(
    path: Path,
    *,
    recommendations: list[dict[str, str]],
    generated_at: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": generated_at if generated_at is not None else time.time(),
        "overall_success_rate": 0.143,
        "routing_recommendations": recommendations,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_avoid_recommendation_swaps_claude_tier_model(tmp_path: Path) -> None:
    _write_hints(
        tmp_path / ".sdd" / "evolution" / "routing_hints.json",
        recommendations=[
            {"role": "backend", "model": "opus", "recommendation": "avoid"},
        ],
    )
    result = _apply_routing_hints(
        workdir=tmp_path,
        role="backend",
        current_model="opus",
        role_model_policy={},
    )
    # Any of the remaining Claude tiers is fine - just not opus.
    assert result != "opus"
    assert result in {"sonnet", "haiku"}


def test_prefer_recommendation_wins_over_default_fallback(tmp_path: Path) -> None:
    _write_hints(
        tmp_path / ".sdd" / "evolution" / "routing_hints.json",
        recommendations=[
            {"role": "backend", "model": "opus", "recommendation": "avoid"},
            {"role": "backend", "model": "sonnet", "recommendation": "prefer"},
        ],
    )
    result = _apply_routing_hints(
        workdir=tmp_path,
        role="backend",
        current_model="opus",
        role_model_policy={},
    )
    assert result == "sonnet"


def test_missing_hints_file_returns_current_model(tmp_path: Path) -> None:
    result = _apply_routing_hints(
        workdir=tmp_path,
        role="backend",
        current_model="opus",
        role_model_policy={},
    )
    assert result == "opus"


def test_stale_hints_file_ignored(tmp_path: Path) -> None:
    stale_ts = time.time() - (48 * 3600)
    _write_hints(
        tmp_path / ".sdd" / "evolution" / "routing_hints.json",
        recommendations=[
            {"role": "backend", "model": "opus", "recommendation": "avoid"},
        ],
        generated_at=stale_ts,
    )
    result = _apply_routing_hints(
        workdir=tmp_path,
        role="backend",
        current_model="opus",
        role_model_policy={},
        max_age_seconds=24 * 3600.0,
    )
    assert result == "opus"


def test_avoid_without_viable_fallback_keeps_current_model(tmp_path: Path) -> None:
    # Non-Claude model, no role_model_policy default -> no fallback candidate.
    _write_hints(
        tmp_path / ".sdd" / "evolution" / "routing_hints.json",
        recommendations=[
            {"role": "backend", "model": "MiniMax-M3", "recommendation": "avoid"},
        ],
    )
    result = _apply_routing_hints(
        workdir=tmp_path,
        role="backend",
        current_model="MiniMax-M3",
        role_model_policy={},
    )
    assert result == "MiniMax-M3"


def test_non_claude_falls_back_to_role_policy_default(tmp_path: Path) -> None:
    _write_hints(
        tmp_path / ".sdd" / "evolution" / "routing_hints.json",
        recommendations=[
            {"role": "backend", "model": "MiniMax-M3", "recommendation": "avoid"},
        ],
    )
    result = _apply_routing_hints(
        workdir=tmp_path,
        role="backend",
        current_model="MiniMax-M3",
        role_model_policy={"default": {"model": "MiniMax-M2.7-highspeed"}},
    )
    assert result == "MiniMax-M2.7-highspeed"


def test_role_without_hints_is_untouched(tmp_path: Path) -> None:
    _write_hints(
        tmp_path / ".sdd" / "evolution" / "routing_hints.json",
        recommendations=[
            {"role": "backend", "model": "opus", "recommendation": "avoid"},
        ],
    )
    result = _apply_routing_hints(
        workdir=tmp_path,
        role="qa",
        current_model="opus",
        role_model_policy={},
    )
    assert result == "opus"
