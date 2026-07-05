"""Shared CI-fix model/effort escalation ladder.

Used by both :mod:`bernstein.github_app.ci_router` and
:mod:`bernstein.gitlab_app.ci_router` so the two CI routing surfaces share
one source of truth for "how many retries before we hand the fix task to a
more capable model."

Configuration (optional, env-var driven -- no yaml parser dependency needed
here since the schema is a small list of triples):

``BERNSTEIN_CI_MODEL_LADDER``: JSON list of ``[retry_threshold, model,
effort]`` triples, evaluated in order, highest threshold that
``retry_count >= retry_threshold`` wins. Example::

    export BERNSTEIN_CI_MODEL_LADDER='[[0,"sonnet","high"],[2,"opus","max"],[4,"opus","max"]]'

If unset, or malformed, falls back to the historical hardcoded default:
``retry_count >= 2 -> opus/max``, else ``sonnet/high``.
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

# Historical hardcoded default, preserved as the fallback ladder so that
# behavior is unchanged when no override is configured.
_DEFAULT_LADDER: list[tuple[int, str, str]] = [
    (0, "sonnet", "high"),
    (2, "opus", "max"),
]

_ENV_VAR = "BERNSTEIN_CI_MODEL_LADDER"


def _parse_ladder_env(raw: str) -> list[tuple[int, str, str]] | None:
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "%s is set but not valid JSON (%s); falling back to default CI "
            "escalation ladder %s. Raw value: %r",
            _ENV_VAR,
            exc,
            _DEFAULT_LADDER,
            raw,
        )
        return None

    if not isinstance(parsed, list) or not parsed:
        logger.warning(
            "%s must be a non-empty JSON list of [retry_threshold, model, "
            "effort] triples; got %r. Falling back to default CI escalation "
            "ladder %s.",
            _ENV_VAR,
            parsed,
            _DEFAULT_LADDER,
        )
        return None

    ladder: list[tuple[int, str, str]] = []
    for entry in parsed:
        if (
            isinstance(entry, list)
            and len(entry) == 3
            and isinstance(entry[0], int)
            and isinstance(entry[1], str)
            and isinstance(entry[2], str)
        ):
            ladder.append((entry[0], entry[1], entry[2]))
        else:
            logger.warning(
                "%s contains a malformed entry %r (expected [int, str, "
                "str]); falling back to default CI escalation ladder %s.",
                _ENV_VAR,
                entry,
                _DEFAULT_LADDER,
            )
            return None

    return sorted(ladder, key=lambda t: t[0])


def resolve_ci_escalation(retry_count: int) -> tuple[str, str]:
    """Return ``(model, effort)`` for a CI-fix task at *retry_count*.

    Reads the ladder from ``BERNSTEIN_CI_MODEL_LADDER`` (JSON list of
    ``[retry_threshold, model, effort]``) if set and valid; otherwise uses
    the historical default (attempts 1-2: sonnet/high, attempts 3+:
    opus/max), preserving prior hardcoded behavior.
    """
    raw = os.environ.get(_ENV_VAR)
    ladder = _DEFAULT_LADDER
    if raw:
        parsed_ladder = _parse_ladder_env(raw)
        if parsed_ladder is not None:
            ladder = parsed_ladder
        else:
            logger.warning(
                "resolve_ci_escalation: using DEFAULT ladder %s for "
                "retry_count=%d because %s failed validation (see prior "
                "warning for details).",
                _DEFAULT_LADDER,
                retry_count,
                _ENV_VAR,
            )

    model, effort = ladder[0][1], ladder[0][2]
    for threshold, ladder_model, ladder_effort in ladder:
        if retry_count >= threshold:
            model, effort = ladder_model, ladder_effort
        else:
            break

    return model, effort
