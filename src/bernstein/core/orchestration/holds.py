"""Orchestrator hold/release registry.

Supplements the orchestrator's quiescence settle-timer self-stop logic
(``open_tasks == 0 and active_agents == 0``) with an explicit "hold" primitive
so external callers (dashboards, long-running human-in-the-loop workflows,
external schedulers) can prevent the orchestrator from self-stopping even
when it looks idle.

A caller acquires a :class:`Hold` via ``acquire_hold(reason, ttl_seconds)``,
does whatever work needs the orchestrator to stay up, then calls
``release_hold(hold_id)`` when done. Holds also expire automatically after
their TTL so a caller that crashes or forgets to release doesn't wedge the
orchestrator open forever.

Thread-safe: backed by a single ``threading.Lock`` since this registry is
read/written from both the FastAPI request-handling threads (via
``orchestrator_holds`` routes) and the orchestrator's own tick loop (via
``tick_pipeline.fetch_active_holds`` -> HTTP -> this module, or in-process
callers that import the singleton directly).
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS: float = 300.0


@dataclass(frozen=True, slots=True)
class Hold(object):
    """A single active hold preventing orchestrator self-stop.

    Attributes:
        id: uuid4 hex identifier for this hold.
        reason: Human-readable reason the hold was acquired (surfaced in logs
            and in the "skipping self-stop" orchestrator message).
        created_at: Epoch seconds when the hold was acquired.
        ttl_seconds: How long the hold stays active before auto-expiring.
        expires_at: Epoch seconds when the hold expires (created_at + ttl_seconds).
    """

    id: str
    reason: str
    created_at: float
    ttl_seconds: float = DEFAULT_TTL_SECONDS
    expires_at: float = field(default=0.0)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dict for API responses."""
        return {
            "id": self.id,
            "reason": self.reason,
            "created_at": self.created_at,
            "ttl_seconds": self.ttl_seconds,
            "expires_at": self.expires_at,
        }


def _make_hold(reason: str, ttl_seconds: float) -> Hold:
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
            ttl_seconds: Auto-expiry window; defaults to 5 minutes so a caller
                that crashes without releasing doesn't wedge the orchestrator
                open indefinitely.

        Returns:
            The newly created Hold.
        """
        hold = _make_hold(reason, ttl_seconds)
        with self._lock:
            self._holds[hold.id] = hold
        logger.info(
            "HoldRegistry.acquire: id=%s reason=%r ttl_seconds=%.1f expires_at=%.1f (active_count=%d)",
            hold.id,
            reason,
            ttl_seconds,
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
            logger.warning("HoldRegistry.release: hold_id=%s not found (already released or expired?)", hold_id)
            return False
        logger.info(
            "HoldRegistry.release: id=%s reason=%r (held for %.1fs)",
            hold_id,
            hold.reason,
            time.time() - hold.created_at,
        )
        return True

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
                        expired.reason,
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


def list_active_holds() -> list[Hold]:
    """List active holds on the module-level singleton registry."""
    return _registry.list_active()


def has_active_holds() -> bool:
    """True if the module-level singleton registry has any active hold."""
    return _registry.has_active()
