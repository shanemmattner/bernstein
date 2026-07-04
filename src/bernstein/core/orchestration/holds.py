"""Orchestrator hold/release registry.

Supplements the orchestrator's quiescence settle-timer self-stop logic
(``open_tasks == 0 and active_agents == 0``) with an explicit "hold" primitive
so external callers (dashboards, long-running human-in-the-loop workflows,
external schedulers) can prevent the orchestrator from self-stopping even
when it looks idle.

A caller acquires a :class:`Hold` via ``acquire_hold(reason, ttl_seconds)``,
then periodically calls ``renew_hold(hold_id)`` (a heartbeat) for as long as
it needs the orchestrator to stay up, then calls ``release_hold(hold_id)``
when done. ``ttl_seconds`` is a grace window, not a run-duration estimate:
each renewal pushes ``expires_at`` out by another ``ttl_seconds`` from "now".
A caller that crashes or stops heartbeating has its hold auto-expire once the
grace window elapses since the last renewal (or since acquisition, if never
renewed) - so it doesn't wedge the orchestrator open forever.

Thread-safe: backed by a single ``threading.Lock`` since this registry is
read/written from both the FastAPI request-handling threads (via
``orchestrator_holds`` routes) and the orchestrator's own tick loop (via
``tick_pipeline.fetch_active_holds`` -> HTTP -> this module, or in-process
callers that import the singleton directly).
"""

from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass, field, replace

from bernstein.core.defaults import HOLDS
from bernstein.core.security.sanitize import sanitize_log

logger = logging.getLogger(__name__)

# Grace-window semantics: as of the heartbeat-renewal model, holds are no
# longer sized to a caller's estimated run duration. A driver acquires a hold
# once, then calls HoldRegistry.renew(hold_id) periodically (heartbeat) to
# keep it alive. DEFAULT_TTL_SECONDS is now the GRACE WINDOW: how long the
# hold survives after the *last* renewal (or after acquire, if never renewed)
# before it is considered abandoned and expires. A short grace window means a
# crashed/hung driver stops blocking orchestrator self-stop quickly, while a
# live driver just needs to heartbeat more often than this window.
#
# Value lives in ``bernstein.core.defaults.HoldsDefaults`` (tunable via the
# ``tuning.holds`` section of bernstein.yaml); re-exported here as a module
# constant for back-compat with existing importers of
# ``holds.DEFAULT_TTL_SECONDS``.
DEFAULT_TTL_SECONDS: float = HOLDS.default_ttl_seconds

# Hard safety bounds on ttl_seconds, enforced in ``_make_hold`` regardless of
# caller: the HTTP request body validation in
# ``routes/orchestrator_holds.py`` is not the only entry point - in-process
# callers that import ``acquire_hold`` directly bypass Pydantic entirely. A
# ttl of 0/negative would expire the hold before (or as soon as) it's used;
# an absurdly large ttl (e.g. 1e12) would defeat this module's documented
# "can't wedge the orchestrator open forever" guarantee. Tunable via
# ``tuning.holds.min_ttl_seconds`` / ``tuning.holds.max_ttl_seconds``.
MIN_TTL_SECONDS: float = HOLDS.min_ttl_seconds
MAX_TTL_SECONDS: float = HOLDS.max_ttl_seconds


@dataclass(frozen=True, slots=True)
class Hold:
    """A single active hold preventing orchestrator self-stop.

    Attributes:
        id: uuid4 hex identifier for this hold.
        reason: Human-readable reason the hold was acquired (surfaced in logs
            and in the "skipping self-stop" orchestrator message).
        created_at: Epoch seconds when the hold was acquired.
        ttl_seconds: Grace window - how long the hold survives after the last
            heartbeat renewal (or since creation, if never renewed) before
            auto-expiring.
        expires_at: Epoch seconds when the hold expires (created_at + ttl_seconds,
            or last_renewed_at + ttl_seconds after a renewal).
        last_renewed_at: Epoch seconds of the most recent heartbeat renewal, or
            None if the hold has never been renewed since acquisition.
    """

    id: str
    reason: str
    created_at: float
    ttl_seconds: float = DEFAULT_TTL_SECONDS
    expires_at: float = field(default=0.0)
    last_renewed_at: float | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dict for API responses."""
        return {
            "id": self.id,
            "reason": self.reason,
            "created_at": self.created_at,
            "ttl_seconds": self.ttl_seconds,
            "expires_at": self.expires_at,
            "last_renewed_at": self.last_renewed_at,
        }


def _make_hold(reason: str, ttl_seconds: float) -> Hold:
    """Build a new :class:`Hold`, clamping ``ttl_seconds`` to safe bounds.

    Callers (HTTP route, in-process ``acquire_hold``) may pass any float,
    including ``0``, negative, ``NaN``, or absurdly large values. Clamping
    here - rather than only at the Pydantic/HTTP boundary - is what makes the
    module's "can't wedge the orchestrator open forever" guarantee actually
    hold for in-process callers too.
    """
    if math.isnan(ttl_seconds) or ttl_seconds < MIN_TTL_SECONDS:
        clamped = MIN_TTL_SECONDS
        logger.warning(
            "HoldRegistry: ttl_seconds=%r is below MIN_TTL_SECONDS=%.1f (or NaN) - clamping to %.1f",
            ttl_seconds,
            MIN_TTL_SECONDS,
            clamped,
        )
        ttl_seconds = clamped
    elif ttl_seconds > MAX_TTL_SECONDS:
        clamped = MAX_TTL_SECONDS
        logger.warning(
            "HoldRegistry: ttl_seconds=%.1f exceeds MAX_TTL_SECONDS=%.1f - clamping to %.1f",
            ttl_seconds,
            MAX_TTL_SECONDS,
            clamped,
        )
        ttl_seconds = clamped

    created_at = time.time()
    return Hold(
        id=uuid.uuid4().hex,
        reason=reason,
        created_at=created_at,
        ttl_seconds=ttl_seconds,
        expires_at=created_at + ttl_seconds,
    )


class HoldRegistry:
    """Thread-safe registry of active orchestrator holds."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._holds: dict[str, Hold] = {}
        logger.info("HoldRegistry initialized")

    def acquire(self, reason: str, ttl_seconds: float = DEFAULT_TTL_SECONDS) -> Hold:
        """Create and store a new hold.

        Args:
            reason: Why the caller wants the orchestrator to stay up.
            ttl_seconds: Grace-window auto-expiry; defaults to
                DEFAULT_TTL_SECONDS (45s) so a caller that crashes without
                releasing (and without heartbeating via renew()) doesn't wedge
                the orchestrator open indefinitely. Clamped to
                [MIN_TTL_SECONDS, MAX_TTL_SECONDS] by ``_make_hold``.

        Returns:
            The newly created Hold (its ``ttl_seconds`` may differ from the
            argument if it was out of bounds and got clamped).
        """
        hold = _make_hold(reason, ttl_seconds)
        with self._lock:
            self._holds[hold.id] = hold
        logger.info(
            "HoldRegistry.acquire: id=%s reason=%r ttl_seconds=%.1f expires_at=%.1f (active_count=%d)",
            hold.id,
            sanitize_log(reason),
            hold.ttl_seconds,
            hold.expires_at,
            len(self._holds),
        )
        return hold

    def release(self, hold_id: str) -> bool:
        """Remove a hold by id.

        Args:
            hold_id: The id of the hold to release.

        Returns:
            True if a hold with that id was found and removed, False otherwise.
        """
        with self._lock:
            hold = self._holds.pop(hold_id, None)
        if hold is None:
            logger.warning(
                "HoldRegistry.release: hold_id=%s not found (already released or expired?)", sanitize_log(hold_id)
            )
            return False
        logger.info(
            "HoldRegistry.release: id=%s reason=%r (held for %.1fs)",
            sanitize_log(hold_id),
            sanitize_log(hold.reason),
            time.time() - hold.created_at,
        )
        return True

    def renew(self, hold_id: str) -> Hold | None:
        """Heartbeat-renew a hold, pushing its expiry out by another grace window.

        Computes and returns the renewed :class:`Hold` under the SAME lock
        acquisition used to do the renewal - callers must not do a separate
        ``get(hold_id)`` afterward to fetch the "current" hold. A second
        lookup races a concurrent ``release(hold_id)``: renew succeeds, then
        the follow-up get 404s because the hold was released in between. This
        also halves the lock acquisitions on this hot ~15s heartbeat path.

        Args:
            hold_id: The id of the hold to renew.

        Returns:
            The renewed Hold if it was found and not already expired, or
            None if the hold is unknown or has already expired.
        """
        now = time.time()
        with self._lock:
            hold = self._holds.get(hold_id)
            if hold is None:
                logger.warning(
                    "HoldRegistry.renew: hold_id=%s not found (never existed, already released, or already expired)",
                    sanitize_log(hold_id),
                )
                return None
            if hold.expires_at < now:
                # Already expired but not yet purged by list_active(); treat as gone.
                self._holds.pop(hold_id, None)
                logger.warning(
                    "HoldRegistry.renew: hold_id=%s found but already expired at %.1f (now=%.1f) - dropping",
                    sanitize_log(hold_id),
                    hold.expires_at,
                    now,
                )
                return None
            new_expires_at = now + hold.ttl_seconds
            renewed = replace(hold, expires_at=new_expires_at, last_renewed_at=now)
            self._holds[hold_id] = renewed
        logger.info(
            "hold %s renewed, new expires_at=%.1f (ttl_seconds=%.1f, last_renewed_at=%.1f)",
            sanitize_log(hold_id),
            new_expires_at,
            renewed.ttl_seconds,
            now,
        )
        return renewed

    def get(self, hold_id: str) -> Hold | None:
        """Look up a single hold by id without purging expired entries.

        Returns:
            The Hold if present (even if technically expired but not yet
            purged), or None if it was never registered / already released.
        """
        with self._lock:
            hold = self._holds.get(hold_id)
        logger.info("HoldRegistry.get: hold_id=%s found=%s", sanitize_log(hold_id), hold is not None)
        return hold

    def list_active(self) -> list[Hold]:
        """Purge expired holds and return the remaining active ones.

        Returns:
            List of Hold objects that have not yet expired.
        """
        now = time.time()
        with self._lock:
            expired_ids = [hid for hid, h in self._holds.items() if h.expires_at < now]
            for hid in expired_ids:
                expired = self._holds.pop(hid, None)
                if expired is not None:
                    logger.info(
                        "HoldRegistry: hold id=%s reason=%r expired at %.1f (ttl_seconds=%.1f)",
                        expired.id,
                        sanitize_log(expired.reason),
                        expired.expires_at,
                        expired.ttl_seconds,
                    )
            active = list(self._holds.values())
        return active

    def has_active(self) -> bool:
        """Convenience wrapper: True if any non-expired hold exists."""
        return len(self.list_active()) > 0


# ---------------------------------------------------------------------------
# Module-level singleton + convenience functions
# ---------------------------------------------------------------------------

_registry = HoldRegistry()


def acquire_hold(reason: str, ttl_seconds: float = DEFAULT_TTL_SECONDS) -> Hold:
    """Acquire a hold on the module-level singleton registry."""
    return _registry.acquire(reason, ttl_seconds)


def release_hold(hold_id: str) -> bool:
    """Release a hold on the module-level singleton registry."""
    return _registry.release(hold_id)


def renew_hold(hold_id: str) -> Hold | None:
    """Heartbeat-renew a hold on the module-level singleton registry.

    Returns the renewed Hold, or None if it was unknown/already expired.
    """
    return _registry.renew(hold_id)


def get_hold(hold_id: str) -> Hold | None:
    """Look up a single hold on the module-level singleton registry."""
    return _registry.get(hold_id)


def list_active_holds() -> list[Hold]:
    """List active holds on the module-level singleton registry."""
    return _registry.list_active()


def has_active_holds() -> bool:
    """True if the module-level singleton registry has any active hold."""
    return _registry.has_active()
