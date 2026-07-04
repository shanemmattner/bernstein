"""Task-level "council of agents" runner.

Unlike the retired ``bernstein.adapters.council_model.CouncilModel`` (a
per-turn ``agents.Model`` subclass that fanned out EVERY individual LLM
call inside a single agent's tool-use loop - far too fine-grained and
expensive), this module runs the council exactly ONCE per task/run: each
candidate model independently drives the ENTIRE task to completion (its
own full multi-turn ``Runner.run`` session, with its own tool calls), and
only then does a judge model see the candidates' final outputs and
synthesize one improved answer from them.

Entry point: :func:`run_council`, called from
``bernstein.adapters.openai_agents_runner._run_session`` exactly where a
plain single-model run would otherwise call ``Runner.run_sync(agent,
prompt)`` - see the ``if manifest.council:`` branch there. The manifest's
``council`` dict is produced by
``openai_agents_runner._load_council_config`` when a role's
``model_field`` value in ``role_model_policy`` ends in ``.yaml``/``.yml``
(that file is parsed into the same
``{"candidates": [...], "judge": {...}, "timeout": <float>}`` shape
documented on :class:`bernstein.core.config.config_schema.CouncilConfig`).

Per the repo's logging house rule ("log everything, prune later"), every
decision point here is logged: which candidate is dispatched and to what
endpoint, each candidate's FULL output (never truncated) before the judge
ever sees it, which candidates timed out/errored (and why) with the
remaining candidates continuing unaffected, the judge's full synthesis
input and output, and the aggregated cost accounting handed back to the
caller.

This module is only ever imported lazily from inside
``openai_agents_runner._run_session``'s ``manifest.council is not None``
branch - by that point the caller has already confirmed ``import agents``
succeeds (see that module's own lazy-import comment), so importing the
``agents``/``openai`` SDKs unconditionally inside the functions below is
safe despite this codebase's usual lazy-SDK-import convention.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bernstein.adapters.openai_agents_runner import RunnerManifest

logger = logging.getLogger(__name__)

# Default per-candidate (and per-judge) wall-clock budget when the council
# config does not set its own ``timeout``. Matches
# ``config_schema.CouncilConfig.timeout``'s Pydantic default.
DEFAULT_TIMEOUT_SECONDS = 120.0


@dataclasses.dataclass
class CouncilRunResult:
    """Minimal duck-typed stand-in for an SDK ``RunResult``.

    ``openai_agents_runner._run_session`` only ever touches two things on
    whatever ``Runner.run_sync``/``run_council`` hands back: ``final_output``
    (for the ``completion`` event's ``summary`` field) and usage data via
    ``_extract_usage_tokens`` - which reads ``result.usage`` first, then
    falls back to summing ``usage`` off every entry of ``result.raw_responses``
    (see that function's docstring: the real SDK ``RunResult`` never
    actually populates ``.usage``, so the ``raw_responses`` fallback is the
    path that fires in practice). Reproducing the SDK's full ``RunResult``
    dataclass (20+ fields, several private/internal) would be both fragile
    and unnecessary - this object only needs to satisfy the two attributes
    those two call sites actually read.

    Attributes:
        final_output: The judge's synthesized text - the council's answer
            for this task/run.
        raw_responses: Every live candidate's ``RunResult.raw_responses``
            entries concatenated with the judge's own, so
            ``_extract_usage_tokens``'s existing fallback path sums real
            token usage across the WHOLE council (every candidate that
            actually ran, plus the judge), not just the winner's slice.
    """

    final_output: str
    raw_responses: list[Any] = dataclasses.field(default_factory=list)


def _resolve_member_client_kwargs(member: dict[str, Any]) -> dict[str, Any]:
    """Delegate to the runner's existing per-member endpoint resolver.

    Imported lazily (rather than at module level) to avoid a circular
    import: ``openai_agents_runner`` imports :func:`run_council` from this
    module, so this module cannot import back from
    ``openai_agents_runner`` at load time.

    Raises:
        RuntimeError: ``api_key_env`` is set but fails validation, or the
            named environment variable is missing/empty - see
            ``openai_agents_runner._resolve_council_member_client_kwargs``.
    """
    from bernstein.adapters.openai_agents_runner import (
        _resolve_council_member_client_kwargs,
    )

    return _resolve_council_member_client_kwargs(member)


async def _run_candidate(
    index: int,
    member: dict[str, Any],
    agent: Any,
    prompt: str,
    timeout: float,
    max_turns: int | None,
) -> tuple[str, Any] | None:
    """Run one candidate's FULL task to completion, independently.

    Returns:
        ``(label, RunResult)`` on success, or ``None`` when the candidate
        timed out or raised - the caller excludes it from judging but
        continues with whatever candidates DID survive.
    """
    from agents import OpenAIChatCompletionsModel, Runner  # type: ignore[import-not-found]
    from openai import AsyncOpenAI  # type: ignore[import-not-found]

    model_id = member.get("model")
    label = f"candidates[{index}]={model_id}"
    client_kwargs = _resolve_member_client_kwargs(member)
    client = AsyncOpenAI(**client_kwargs)

    candidate_model = OpenAIChatCompletionsModel(model=model_id, openai_client=client)
    candidate_agent = agent.clone(model=candidate_model)

    run_kwargs: dict[str, Any] = {} if max_turns is None else {"max_turns": max_turns}
    logger.info(
        "run_council._run_candidate: dispatching %s base_url=%r (api_key_env=%r) "
        "timeout=%.1fs max_turns=%s",
        label,
        member.get("base_url") or "<default>",
        member.get("api_key_env") or "<ambient OPENAI_API_KEY>",
        timeout,
        max_turns if max_turns is not None else "SDK-default",
    )
    started = time.monotonic()
    try:
        result = await asyncio.wait_for(
            Runner.run(candidate_agent, prompt, **run_kwargs),
            timeout=timeout,
        )
    except TimeoutError:
        logger.warning(
            "run_council._run_candidate: %s TIMED OUT after %.1fs - excluding it from "
            "judge synthesis, continuing with remaining candidates",
            label,
            timeout,
        )
        return None
    except Exception as exc:  # noqa: BLE001 - one bad candidate must never kill the council
        logger.warning(
            "run_council._run_candidate: %s FAILED after %.2fs: %s: %s - excluding it "
            "from judge synthesis, continuing with remaining candidates",
            label,
            time.monotonic() - started,
            type(exc).__name__,
            exc,
        )
        return None

    elapsed = time.monotonic() - started
    output_text = str(getattr(result, "final_output", ""))
    logger.info(
        "run_council._run_candidate: %s responded in %.2fs - FULL output follows:\n%s",
        label,
        elapsed,
        output_text,
    )
    return label, result


async def _run_judge(
    judge_member: dict[str, Any],
    task_prompt: str,
    candidate_outputs: list[tuple[str, str]],
    timeout: float,
    agent: Any,
    max_turns: int | None,
) -> Any:
    """Ask the judge model to SYNTHESIZE one improved answer from every candidate.

    Unlike the retired ``CouncilModel``, which restricted its judge to
    picking one candidate's response verbatim (see that module's
    docstring for why - tool-call structural risk at the per-turn level),
    a task-level judge only ever sees plain TEXT (each candidate's final
    output after its own full run completed), so there is no structured
    tool-call payload to corrupt. The judge is explicitly instructed to
    combine the best insights from every candidate and correct any errors
    it spots, not to just pick a winner.

    Args:
        judge_member: The council config's ``judge`` entry
            (``{"model": ..., "base_url": ..., "api_key_env": ...}``).
        task_prompt: The original task prompt, included so the judge can
            evaluate candidate outputs against the actual task.
        candidate_outputs: ``(label, full_output_text)`` for every candidate
            that survived (i.e. did not time out or error).
        timeout: Wall-clock budget for the judge call itself.
        agent: The base agent definition (used only for ``.clone()`` so the
            judge inherits the same ``name``/``instructions`` prefix
            conventions as the candidates; its ``tools``/``handoffs`` are
            stripped since the judge only ever produces text).

    Returns:
        The judge's ``RunResult``.
    """
    from agents import OpenAIChatCompletionsModel, Runner  # type: ignore[import-not-found]
    from openai import AsyncOpenAI  # type: ignore[import-not-found]

    model_id = judge_member.get("model")
    client_kwargs = _resolve_member_client_kwargs(judge_member)
    client = AsyncOpenAI(**client_kwargs)
    judge_model = OpenAIChatCompletionsModel(model=model_id, openai_client=client)
    # Strip tools/handoffs: the judge only ever reads text and produces
    # text - it must never attempt to call a tool the original task agent
    # had access to.
    judge_agent = agent.clone(model=judge_model, tools=[], handoffs=[])

    candidate_sections = "\n\n".join(
        f"### Candidate: {label}\n{output_text}" for label, output_text in candidate_outputs
    )
    judge_prompt = (
        "You are synthesizing the best answer from multiple AI agents who each "
        "analyzed the same task independently. Each candidate below completed "
        "the ENTIRE task on its own (its own full multi-step run) and produced "
        "a final output.\n\n"
        f"## Original task\n{task_prompt}\n\n"
        f"## Candidate outputs\n{candidate_sections}\n\n"
        "## Your job\n"
        "Combine the best insights, approaches, and correctness from EVERY "
        "candidate above into ONE unified, improved response. Correct any "
        "errors you notice in any candidate's output. Do NOT simply pick one "
        "candidate's answer verbatim and repeat it - genuinely synthesize a "
        "response that is better than any single candidate's output on its "
        "own. Output only the final synthesized response - no meta-commentary "
        "about the synthesis process, no mention of 'candidate 1/2/3'."
    )

    run_kwargs: dict[str, Any] = {} if max_turns is None else {"max_turns": max_turns}
    logger.info(
        "run_council._run_judge: dispatching judge model=%r base_url=%r (api_key_env=%r) "
        "timeout=%.1fs max_turns=%s over %d candidate output(s) - full judge prompt follows:\n%s",
        model_id,
        judge_member.get("base_url") or "<default>",
        judge_member.get("api_key_env") or "<ambient OPENAI_API_KEY>",
        timeout,
        max_turns if max_turns is not None else "SDK-default",
        len(candidate_outputs),
        judge_prompt,
    )
    started = time.monotonic()
    judge_result = await asyncio.wait_for(
        Runner.run(judge_agent, judge_prompt, **run_kwargs),
        timeout=timeout,
    )
    elapsed = time.monotonic() - started
    output_text = str(getattr(judge_result, "final_output", ""))
    logger.info(
        "run_council._run_judge: judge responded in %.2fs - FULL synthesis output follows:\n%s",
        elapsed,
        output_text,
    )
    return judge_result


async def _run_council_async(
    agent: Any,
    prompt: str,
    council_cfg: dict[str, Any],
    manifest: RunnerManifest,
) -> CouncilRunResult:
    """Async implementation backing :func:`run_council` - see its docstring."""
    started = time.monotonic()
    raw_candidates = council_cfg.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        msg = "run_council: council_cfg.candidates must be a non-empty list"
        logger.error(msg)
        raise RuntimeError(msg)
    judge_member = council_cfg.get("judge")
    if not isinstance(judge_member, dict):
        msg = "run_council: council_cfg.judge is required and must be a mapping"
        logger.error(msg)
        raise RuntimeError(msg)
    timeout_value = council_cfg.get("timeout")
    timeout = float(timeout_value) if isinstance(timeout_value, (int, float)) else DEFAULT_TIMEOUT_SECONDS

    # Reuse the runner's own max_turns resolution (manifest field > env var >
    # yaml tuning default > SDK default) so a council role gets the exact
    # same turn-cap behavior a single-model role would - imported lazily to
    # avoid a circular import at module load (see module docstring).
    from bernstein.adapters.openai_agents_runner import _resolve_max_turns

    max_turns = _resolve_max_turns(manifest.max_turns)

    logger.info(
        "run_council: session=%s ENTER - task-level council run starting: %d candidates, "
        "timeout=%.1fs per candidate/judge call, max_turns=%s",
        manifest.session_id,
        len(raw_candidates),
        timeout,
        max_turns if max_turns is not None else "SDK-default",
    )

    # Every candidate runs its OWN full task end-to-end, in parallel. One
    # candidate hanging or erroring must never block/kill the others -
    # ``asyncio.gather`` without ``return_exceptions`` would propagate the
    # first exception and cancel every sibling task, so ``_run_candidate``
    # itself catches everything and returns ``None`` for a dead candidate
    # instead of letting the exception escape to ``gather``.
    candidate_results = await asyncio.gather(
        *(
            _run_candidate(i, member, agent, prompt, timeout, max_turns)
            for i, member in enumerate(raw_candidates)
        ),
    )

    live: list[tuple[str, Any]] = [r for r in candidate_results if r is not None]
    if not live:
        msg = (
            f"run_council: session={manifest.session_id}: all {len(raw_candidates)} "
            f"candidates failed or timed out (timeout={timeout}s) - see WARNING logs "
            f"above for each candidate's exact failure. No output exists for the judge "
            f"to synthesize from."
        )
        logger.error(msg)
        raise RuntimeError(msg)

    logger.info(
        "run_council: session=%s: %d/%d candidates survived: %s - proceeding to judge synthesis",
        manifest.session_id,
        len(live),
        len(raw_candidates),
        [label for label, _ in live],
    )

    candidate_outputs = [(label, str(getattr(result, "final_output", ""))) for label, result in live]
    judge_result = await _run_judge(judge_member, prompt, candidate_outputs, timeout, agent, max_turns)

    # Aggregate cost accounting: concatenate every live candidate's
    # ``raw_responses`` with the judge's own so the existing
    # ``_extract_usage_tokens`` fallback (which sums ``.usage`` off each
    # ``raw_responses`` entry - see ``CouncilRunResult`` docstring) prices
    # the WHOLE council's real spend, not just the judge's slice of it.
    aggregated_raw_responses: list[Any] = []
    for _, result in live:
        aggregated_raw_responses.extend(getattr(result, "raw_responses", None) or [])
    aggregated_raw_responses.extend(getattr(judge_result, "raw_responses", None) or [])

    final_output = str(getattr(judge_result, "final_output", ""))
    logger.info(
        "run_council: session=%s EXIT - council run complete in %.2fs: %d/%d candidates "
        "live, %d aggregated raw_responses entries carried forward for usage accounting",
        manifest.session_id,
        time.monotonic() - started,
        len(live),
        len(raw_candidates),
        len(aggregated_raw_responses),
    )
    return CouncilRunResult(final_output=final_output, raw_responses=aggregated_raw_responses)


def run_council(
    agent: Any,
    prompt: str,
    council_cfg: dict[str, Any],
    manifest: RunnerManifest,
) -> CouncilRunResult:
    """Run a task-level council: N candidates run the FULL task in parallel,
    each wrapped in a per-candidate timeout, then a judge synthesizes one
    unified answer from whichever candidates survived.

    Call-site compatible with ``Runner.run_sync(agent, prompt)`` - see the
    ``if manifest.council:`` branch in
    ``openai_agents_runner._run_session`` that calls this instead. The
    returned :class:`CouncilRunResult` duck-types the two attributes that
    caller actually reads off its ``result`` (``final_output`` and, via the
    existing ``_extract_usage_tokens`` fallback, ``raw_responses``).

    Args:
        agent: The base ``agents.Agent`` definition built by
            ``openai_agents_runner._build_agent_kwargs``/``agent_cls(...)``
            (name, instructions, tools). Each candidate/judge gets its own
            ``.clone(model=...)`` off this same definition - only the
            model differs per member.
        prompt: The task prompt, forwarded verbatim to every candidate and
            included in the judge's synthesis prompt.
        council_cfg: Parsed council config -
            ``{"candidates": [{"model", "base_url"?, "api_key_env"?}, ...],
            "judge": {...}, "timeout"?: float}`` - see
            ``bernstein.core.config.config_schema.CouncilConfig``.
        manifest: The parsed ``RunnerManifest`` (used for session-id log
            correlation only).

    Returns:
        A :class:`CouncilRunResult` whose ``final_output`` is the judge's
        synthesized text.

    Raises:
        RuntimeError: ``council_cfg`` is malformed, or every candidate
            failed/timed out (the full per-candidate failure reason is
            logged at WARNING before this is raised).
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # Common case: ``_run_session`` calls this from plain synchronous
        # code with no event loop running in this thread yet - safe to
        # drive our own with ``asyncio.run``.
        logger.info("run_council: no event loop running in this thread - using asyncio.run()")
        return asyncio.run(_run_council_async(agent, prompt, council_cfg, manifest))

    # A loop IS already running in this thread (e.g. pytest-asyncio tests
    # that exercise this module directly) - ``asyncio.run()`` would raise
    # "cannot be called from a running event loop". Drive the whole council
    # coroutine from a dedicated thread with its own fresh loop instead.
    logger.info(
        "run_council: an event loop is already running in this thread - driving the "
        "council coroutine from a dedicated worker thread instead of asyncio.run()"
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, _run_council_async(agent, prompt, council_cfg, manifest))
        return future.result()
