# Orchestrator holds

The orchestrator self-stops on quiescence: when `open_tasks == 0` and
`active_agents == 0` for a settle window, it exits. The hold/release API lets
an external caller (a dashboard, a human-in-the-loop workflow, a scheduler
like `run.py`) prevent that self-stop while it still has work planned, even
though the orchestrator looks idle right now.

A hold is a lightweight, heartbeat-renewed lease, not a fixed-duration
reservation: acquire it before a gap in task submission, heartbeat-renew it
periodically for as long as you need the orchestrator to stay up, then
release it when you're done. `ttl_seconds` is a **grace window**, not a
run-duration estimate — each renewal pushes the hold's expiry out by another
`ttl_seconds` from "now". If the caller crashes or stops heartbeating, the
hold auto-expires once the grace window elapses since the last renewal (or
since acquisition, if never renewed), so the orchestrator can self-stop
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

`ttl_seconds` is optional; the server default is 45s (`DEFAULT_TTL_SECONDS`
in `bernstein.core.orchestration.holds`, tunable via `tuning.holds` in
`bernstein.yaml` — see `bernstein.core.defaults.HoldsDefaults`). This is a
grace window, not a run-duration estimate: see
[TTL and renewal behavior](#ttl-and-renewal-behavior) below. Response:

```json
{
  "id": "3f9c2b1a...",
  "reason": "waiting on human approval for phase 2",
  "created_at": 1751600000.0,
  "ttl_seconds": 300.0,
  "expires_at": 1751600300.0,
  "last_renewed_at": null
}
```

### `POST /orchestrator/holds/{hold_id}/renew`

Heartbeat-renew a hold, pushing its expiry out by another `ttl_seconds` grace
window from now. Callers that need a hold to stay alive for longer than one
`ttl_seconds` window **must** call this periodically — see
[TTL and renewal behavior](#ttl-and-renewal-behavior). Returns the renewed
hold (same shape as the `POST /orchestrator/holds` response above), or `404`
if the hold never existed, was already released, or had already expired
before the renew call landed.

### `DELETE /orchestrator/holds/{hold_id}`

Release a hold by id. Returns `{"released": true}`, or `404` if the hold was
already released or has expired.

### `GET /orchestrator/holds`

List all currently active (non-expired) holds:

```json
{"holds": [...], "count": 1}
```

---

## TTL and renewal behavior

`ttl_seconds` is a **grace window**, not a fixed lease duration — this is the
single most important thing to understand about this API, and it's easy to
miss if you only skim the request/response shapes above.

- A hold expires at `expires_at`, which starts as `created_at + ttl_seconds`
  and is pushed out to `renewed_at + ttl_seconds` every time
  `POST /orchestrator/holds/{hold_id}/renew` succeeds.
- **A caller that needs the orchestrator held open longer than one
  `ttl_seconds` window must heartbeat-renew before it expires** — there is no
  way to acquire a single hold that lasts an arbitrarily long time by passing
  a huge `ttl_seconds` up front (the server clamps `ttl_seconds` to a sane
  range; see `bernstein.core.defaults.HoldsDefaults`). Treat `ttl_seconds` as
  "how long am I allowed to go dark before the orchestrator assumes I died,"
  not "how long will my task take."
- A caller that crashes or simply stops calling `/renew` has its hold expire
  on its own once the grace window elapses since the last renewal (or since
  acquisition, if it never renewed) — so a dead driver doesn't wedge the
  orchestrator open forever.
- Expiry is lazy: a hold is purged the next time `list_active()` runs (every
  quiescence check and every `GET` call), not on a background timer.
- While at least one hold is active, the orchestrator's quiescence
  self-stop check is skipped for that tick and logged as
  `"Quiescence detected but N active hold(s) present ... skipping self-stop"`.
- The hold check fails open: if the orchestrator cannot fetch the hold list
  (endpoint unreachable, malformed response), it logs a warning and treats
  the tick as having no active holds. A hold is advisory, never a hard lock
  that can wedge shutdown.

## Usage pattern

A driver script that submits work in phases, with a gap between phases where
no tasks exist yet, heartbeating every third of the TTL so it never lapses:

```python
hold = acquire_hold("phase-2 prep", ttl_seconds=600)
try:
    while waiting_on_phase_2_decision():
        # ... do phase-1 wrap-up, decide phase-2 tasks ...
        time.sleep(200)  # well under the 600s grace window
        renew_hold(hold.id)
    submit_phase_2_tasks()
finally:
    release_hold(hold.id)
```

Backward compatibility: `BERNSTEIN_QUIESCENCE_SETTLE_S` still applies for
runs that never acquire a hold - this API is additive, not a replacement for
non-driven runs.

## Code pointers

| File | What it does |
|------|--------------|
| `src/bernstein/core/orchestration/holds.py` | `HoldRegistry`, `Hold`, module-level singleton |
| `src/bernstein/core/routes/orchestrator_holds.py` | FastAPI routes |
| `src/bernstein/core/orchestration/tick_pipeline.py` | `fetch_active_holds` - consulted before self-stop |
