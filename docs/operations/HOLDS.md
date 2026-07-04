# Orchestrator holds

The orchestrator self-stops on quiescence: when `open_tasks == 0` and
`active_agents == 0` for a settle window, it exits. The hold/release API lets
an external caller (a dashboard, a human-in-the-loop workflow, a scheduler
like `run.py`) prevent that self-stop while it still has work planned, even
though the orchestrator looks idle right now.

A hold is a lightweight, TTL-bounded lease: acquire it before a gap in task
submission, release it when you're done. If the caller crashes or forgets to
release, the hold expires on its own and the orchestrator can self-stop
again.

---

## Endpoints

All endpoints are under `/orchestrator/holds` on the running task server.

### `POST /orchestrator/holds`

Acquire a hold.

```json
{
  "reason": "waiting on human approval for phase 2",
  "ttl_seconds": 300
}
```

`ttl_seconds` is optional; the server default is 300s (`DEFAULT_TTL_SECONDS`
in `bernstein.core.orchestration.holds`). Response:

```json
{
  "id": "3f9c2b1a...",
  "reason": "waiting on human approval for phase 2",
  "created_at": 1751600000.0,
  "ttl_seconds": 300.0,
  "expires_at": 1751600300.0
}
```

### `DELETE /orchestrator/holds/{hold_id}`

Release a hold by id. Returns `{"released": true}`, or `404` if the hold was
already released or has expired.

### `GET /orchestrator/holds`

List all currently active (non-expired) holds:

```json
{"holds": [...], "count": 1}
```

---

## TTL behavior

- Holds expire automatically at `created_at + ttl_seconds`. There is no
  renewal call — a caller that needs a hold to last longer must release and
  re-acquire, or use a larger `ttl_seconds` up front.
- Expiry is lazy: a hold is purged the next time `list_active()` runs (every
  quiescence check and every `GET`/`POST` call), not on a background timer.
- While at least one hold is active, the orchestrator's quiescence
  self-stop check is skipped for that tick and logged as
  `"Quiescence detected but N active hold(s) present ... skipping self-stop"`.

## Usage pattern

A driver script that submits work in phases, with a gap between phases where
no tasks exist yet:

```python
hold = acquire_hold("phase-2 prep", ttl_seconds=600)
try:
    # ... do phase-1 wrap-up, decide phase-2 tasks ...
    submit_phase_2_tasks()
finally:
    release_hold(hold.id)
```

Backward compatibility: `BERNSTEIN_QUIESCENCE_SETTLE_S` still applies for
runs that never acquire a hold — this API is additive, not a replacement for
non-driven runs.

## Code pointers

| File | What it does |
|------|--------------|
| `src/bernstein/core/orchestration/holds.py` | `HoldRegistry`, `Hold`, module-level singleton |
| `src/bernstein/core/routes/orchestrator_holds.py` | FastAPI routes |
| `src/bernstein/core/orchestration/tick_pipeline.py` | `fetch_active_holds` — consulted before self-stop |
