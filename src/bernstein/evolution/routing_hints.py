"""Consumer helpers for ``routing_hints.json`` produced by SuccessRateAdvisor.

The advisor writes recommendations to ``<state_dir>/evolution/routing_hints.json``
so downstream code (e.g. the spawner's role/model resolver) can bias model
selection away from low-success (role, model) pairs and toward high-success
ones.

Producer side lives in :mod:`bernstein.evolution.detector`
(``SuccessRateAdvisor.export_routing_hints``). This module is the read side:
loading the file, exposing a typed view keyed by role, and providing a small
pure-function selector that a caller can plug into their existing model-choice
path without threading the raw JSON around.

Design constraints:
- Missing / malformed / stale files must be treated as "no hints" — never
  raise into the spawn hot path.
- Hints are advisory. A role/model pair that is only *soft-blocked* (avoid)
  with no fallback candidate must fall through to the caller's default,
  never yield an empty string or raise.
- Freshness is measured against ``generated_at`` (epoch seconds) so callers
  can cheaply discard hints from a stopped/crashed evolution loop.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoleRoutingHints:
    """Per-role routing advisory computed from recent task metrics."""

    avoid: frozenset[str] = field(default_factory=frozenset)
    prefer: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class RoutingHints:
    """Loaded view of ``routing_hints.json``."""

    generated_at: float
    overall_success_rate: float
    by_role: dict[str, RoleRoutingHints]

    @property
    def is_empty(self) -> bool:
        return not self.by_role

    def for_role(self, role: str) -> RoleRoutingHints:
        return self.by_role.get(role, RoleRoutingHints())


_EMPTY = RoutingHints(generated_at=0.0, overall_success_rate=0.0, by_role={})


def load_routing_hints(
    path: Path, *, max_age_seconds: float | None = None
) -> RoutingHints:
    """Load routing hints from ``path``. Never raises.

    Args:
        path: Location of ``routing_hints.json`` (typically
            ``<state_dir>/evolution/routing_hints.json``).
        max_age_seconds: If set and the file's ``generated_at`` is older than
            this many seconds, return empty hints (treat as stale).

    Returns:
        A :class:`RoutingHints` view; empty on any error, missing file, or
        stale timestamp.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return _EMPTY
    except OSError:
        logger.exception("Failed to read routing hints from %s", path)
        return _EMPTY

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("routing_hints.json at %s is not valid JSON; ignoring", path)
        return _EMPTY

    if not isinstance(data, dict):
        return _EMPTY

    generated_at = _coerce_float(data.get("generated_at"), default=0.0)
    if max_age_seconds is not None and generated_at > 0.0:
        if time.time() - generated_at > max_age_seconds:
            return _EMPTY

    overall = _coerce_float(data.get("overall_success_rate"), default=0.0)

    by_role: dict[str, dict[str, set[str]]] = {}
    recs = data.get("routing_recommendations")
    if not isinstance(recs, list):
        return RoutingHints(
            generated_at=generated_at,
            overall_success_rate=overall,
            by_role={},
        )

    for entry in recs:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        model = entry.get("model")
        rec = entry.get("recommendation")
        if not (isinstance(role, str) and isinstance(model, str) and isinstance(rec, str)):
            continue
        bucket = by_role.setdefault(role, {"avoid": set(), "prefer": set()})
        if rec == "avoid":
            bucket["avoid"].add(model)
        elif rec == "prefer":
            bucket["prefer"].add(model)

    return RoutingHints(
        generated_at=generated_at,
        overall_success_rate=overall,
        by_role={
            role: RoleRoutingHints(
                avoid=frozenset(buckets["avoid"]),
                prefer=frozenset(buckets["prefer"]),
            )
            for role, buckets in by_role.items()
        },
    )


def select_model_with_hints(
    role: str,
    candidate: str,
    hints: RoutingHints,
    *,
    fallback_candidates: tuple[str, ...] = (),
) -> str:
    """Return the model to use for ``role``, biased by advisor hints.

    Rules, in priority order:

    1. If ``candidate`` is in the role's ``avoid`` set AND a viable fallback
       exists (a fallback candidate that is not itself in ``avoid``, or any
       model in ``prefer``), return that alternative.
    2. Otherwise, return ``candidate`` unchanged — hints are advisory.

    A ``prefer`` alternative wins over a merely-not-avoided fallback.

    Args:
        role: The task role the model is being chosen for.
        candidate: The model the caller would have chosen absent hints.
        hints: Loaded advisor hints.
        fallback_candidates: Ordered list of other models the caller would
            accept for this role (e.g. from ``role_model_policy['default']``).

    Returns:
        The model name to use.
    """
    role_hints = hints.for_role(role)
    if candidate not in role_hints.avoid:
        return candidate

    for preferred in role_hints.prefer:
        if preferred != candidate:
            return preferred

    for alt in fallback_candidates:
        if alt and alt not in role_hints.avoid and alt != candidate:
            return alt

    logger.info(
        "routing_hints: role=%r candidate=%r flagged avoid but no viable "
        "fallback (prefer=%s, fallbacks=%s); keeping candidate",
        role,
        candidate,
        sorted(role_hints.prefer),
        list(fallback_candidates),
    )
    return candidate


def _coerce_float(value: object, *, default: float) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default
