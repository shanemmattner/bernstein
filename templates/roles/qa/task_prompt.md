# Task: {{TASK_TITLE}}

## Description
{{TASK_DESCRIPTION}}

{{#IF FILES}}
## Files to work with
{{FILES}}
{{/IF}}

{{#IF CONTEXT}}
## Context
{{CONTEXT}}
{{/IF}}

## Instructions
1. Read the code under test before writing a single test
2. Cover: happy path, edge cases (empty, boundary, None), and error paths
3. Use descriptive test names: `test_<function>_<scenario>_<expected_outcome>`
4. Mock external dependencies (network, filesystem, time); do NOT mock internal logic
5. Run the full suite to check for regressions: `uv run python scripts/run_tests.py -x`
6. If you find a bug while testing, document it as a failing test before fixing

## Pre-completion checklist (janitor verification will check these)
Before you submit the completion payload, verify ALL of the following. Skipping any of these is the #1 cause of `janitor_failure` for the qa role:
- Your new/updated test file exists on disk at the exact path listed in your task's `owned_files`.
- The targeted suite for your new tests passes with exit code 0: `uv run pytest <your_test_file> -x -q`. Record the exact command and its exit code as the `verification` field of your completion payload.
- The full regression run passes: `uv run python scripts/run_tests.py -x`. If it fails, your task is NOT done — fix the code or, if the failure is unrelated and outside your `owned_files`, use the `blocked_on_dependency` refusal instead of marking complete.
- Your commit is on branch `agent/qa-<id>` (never main, never someone else's branch) and contains only your `owned_files`. Run `git log -1 --stat` to confirm.
- The `files_changed` array in your completion payload lists exactly the files you committed — no more, no less.

## If stuck or blocked
- If a curl to the task server fails with a connection error, retry up to 3 times with 2-second delays. Do NOT retry on 4xx responses — the state has changed and retrying will not help.
- If tests fail after your changes, fix the code. Do not skip tests, mark them xfail, or mark the task complete with failures — the janitor will re-open the task and this counts as `janitor_failure`.
- If you cannot proceed as specified, do NOT invent a workaround and do NOT call `/complete` with a bogus payload. Emit a typed refusal via the completion contract (see below) — `underspecified`, `awaiting_operator`, `scope_exceeded`, or `blocked_on_dependency` — and stop.
- If blocked by another agent's files, post to the bulletin board and move on.

## Bulletin board
Post discoveries, new APIs, or blockers so other parallel agents stay informed:
```bash
curl -sS -w '\n%{http_code}' -X POST {{SERVER_URL}}/bulletin \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "{{AGENT_ID}}", "type": "finding", "content": "<what you created or discovered>"}'
```

{{INCLUDE completion_contract}}
