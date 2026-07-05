"""Opportunity detection from aggregated metrics."""

from __future__ import annotations

import contextlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from bernstein.evolution.aggregator import MetricsCollector, TaskMetrics


class UpgradeCategory(Enum):
    """Category of upgrade."""

    POLICY_UPDATE = "policy_update"  # Low risk policy tweaks
    ROUTING_RULES = "routing_rules"  # Model routing adjustments
    MODEL_ROUTING = "model_routing"  # Model selection changes
    ROLE_TEMPLATES = "role_templates"  # Prompt template updates
    PROVIDER_CONFIG = "provider_config"  # Provider configuration


@dataclass
class ImprovementOpportunity:
    """Identified improvement opportunity."""

    category: UpgradeCategory
    title: str
    description: str
    expected_improvement: str
    confidence: float
    risk_level: Literal["low", "medium", "high"]
    affected_components: list[str] = field(default_factory=list[str])
    estimated_cost_impact_usd: float = 0.0


@dataclass
class FailurePattern:
    """A detected pattern of recurring failures.

    Groups failures by role and error type to identify systematic issues
    that can be addressed through configuration or template changes.
    """

    task_type: str
    error_pattern: str
    occurrence_count: int
    affected_models: list[str]
    first_seen: float
    last_seen: float
    sample_task_ids: list[str] = field(default_factory=list[str])


@dataclass
class FailureRecord:
    """A single failure event for JSONL persistence."""

    timestamp: float
    task_id: str
    role: str
    model: str | None
    error_type: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "timestamp": self.timestamp,
            "task_id": self.task_id,
            "role": self.role,
            "model": self.model,
            "error_type": self.error_type,
        }


class FailureAnalyzer:
    """Tracks and analyzes task failures to detect recurring patterns.

    Persists failure records to `.sdd/evolution/failures.jsonl` and provides
    methods to detect patterns, compute failure rates by role/model, and
    surface actionable insights for the evolution loop.
    """

    def __init__(self, state_dir: Path | None = None) -> None:
        if state_dir is None:
            state_dir = Path(".sdd")
        self._evolution_dir = state_dir / "evolution"
        self._evolution_dir.mkdir(parents=True, exist_ok=True)
        self.failures_path = self._evolution_dir / "failures.jsonl"
        self._failures: list[FailureRecord] = []
        self._load()

    def _load(self) -> None:
        """Load existing failure records from disk."""
        if not self.failures_path.exists():
            return
        with self.failures_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    self._failures.append(
                        FailureRecord(
                            timestamp=data["timestamp"],
                            task_id=data["task_id"],
                            role=data["role"],
                            model=data.get("model"),
                            error_type=data["error_type"],
                        )
                    )
                except (json.JSONDecodeError, KeyError):
                    continue

    def record_failure(
        self,
        task_id: str,
        role: str,
        model: str | None,
        error_type: str,
    ) -> None:
        """Record a task failure to disk and in-memory list.

        Args:
            task_id: Unique identifier of the failed task.
            role: The role that was assigned to the task.
            model: The model used, or None if unknown.
            error_type: Short description of the error category.
        """
        record = FailureRecord(
            timestamp=time.time(),
            task_id=task_id,
            role=role,
            model=model,
            error_type=error_type,
        )
        self._failures.append(record)
        with self.failures_path.open("a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

    def detect_patterns(self, min_occurrences: int = 3) -> list[FailurePattern]:
        """Detect recurring failure patterns by grouping on (role, error_type).

        Args:
            min_occurrences: Minimum number of failures to qualify as a pattern.

        Returns:
            List of failure patterns meeting the occurrence threshold.
        """
        groups: dict[tuple[str, str], list[FailureRecord]] = {}
        for record in self._failures:
            key = (record.role, record.error_type)
            groups.setdefault(key, []).append(record)

        patterns: list[FailurePattern] = []
        for (role, error_type), records in groups.items():
            if len(records) < min_occurrences:
                continue
            models = list({r.model for r in records if r.model is not None})
            task_ids = [r.task_id for r in records[:5]]
            patterns.append(
                FailurePattern(
                    task_type=role,
                    error_pattern=error_type,
                    occurrence_count=len(records),
                    affected_models=models,
                    first_seen=records[0].timestamp,
                    last_seen=records[-1].timestamp,
                    sample_task_ids=task_ids,
                )
            )
        return patterns

    def get_failure_rate_by_role(self, hours: int = 24) -> dict[str, float]:
        """Compute failure rate per role over a recent time window.

        Args:
            hours: Number of hours to look back.

        Returns:
            Mapping of role name to failure rate (0.0-1.0).  The rate is
            computed as failures / total failures across all roles in the
            window, giving a relative distribution.  Returns 1.0 for each
            role since all records here are failures; pair with task metrics
            for absolute rates.
        """
        cutoff = time.time() - (hours * 3600)
        recent = [r for r in self._failures if r.timestamp >= cutoff]
        if not recent:
            return {}

        role_counts: dict[str, int] = {}
        for r in recent:
            role_counts[r.role] = role_counts.get(r.role, 0) + 1

        total = len(recent)
        return {role: count / total for role, count in role_counts.items()}

    def get_failure_rate_by_model(self, hours: int = 24) -> dict[str, float]:
        """Compute failure rate per model over a recent time window.

        Args:
            hours: Number of hours to look back.

        Returns:
            Mapping of model name to failure rate (0.0-1.0).  Same
            distribution semantics as ``get_failure_rate_by_role``.
        """
        cutoff = time.time() - (hours * 3600)
        recent = [r for r in self._failures if r.timestamp >= cutoff]
        if not recent:
            return {}

        model_counts: dict[str, int] = {}
        for r in recent:
            model_name = r.model if r.model is not None else "unknown"
            model_counts[model_name] = model_counts.get(model_name, 0) + 1

        total = len(recent)
        return {model: count / total for model, count in model_counts.items()}


@dataclass
class ModelRouteRecommendation:
    """A specific routing recommendation for a role/model combination."""

    role: str
    model: str
    success_rate: float
    failure_count: int
    recommendation: Literal["avoid", "prefer"]
    reason: str


@dataclass
class SuccessRateAnalysis:
    """Analysis of task success rates broken down by role and model."""

    overall_rate: float
    total_tasks: int
    role_rates: dict[str, float]
    model_rates: dict[str, float]
    routing_recommendations: list[ModelRouteRecommendation]


class SuccessRateAdvisor:
    """Analyzes task success rates and generates model routing recommendations.

    Computes per-role and per-(role, model) success rates from recent task
    metrics, then recommends routing changes to steer tasks away from
    poorly-performing model/role combinations toward high-performing ones.

    The advisor requires at least ``_MIN_SAMPLES_PER_GROUP`` task metrics
    for a (role, model) pair before emitting a recommendation for it, to
    avoid acting on statistical noise.
    """

    _MIN_SAMPLES_PER_GROUP: int = 3
    _AVOID_THRESHOLD: float = 0.5
    _PREFER_THRESHOLD: float = 0.9

    def __init__(
        self,
        collector: MetricsCollector,
    ) -> None:
        self.collector = collector

    def analyze(self, hours: int = 24) -> SuccessRateAnalysis:
        """Compute success rates and routing recommendations from recent metrics.

        Args:
            hours: How far back to look for task metrics.

        Returns:
            A :class:`SuccessRateAnalysis` with per-role/model breakdowns and
            concrete routing recommendations.
        """
        task_metrics: list[TaskMetrics] = self.collector.get_recent_task_metrics(hours=hours)
        if not task_metrics:
            return SuccessRateAnalysis(
                overall_rate=0.0,
                total_tasks=0,
                role_rates={},
                model_rates={},
                routing_recommendations=[],
            )

        overall = sum(1 for m in task_metrics if m.janitor_passed) / len(task_metrics)

        role_groups: dict[str, list[TaskMetrics]] = {}
        for m in task_metrics:
            role = m.role or "unknown"
            role_groups.setdefault(role, []).append(m)

        role_rates: dict[str, float] = {
            role: sum(1 for m in ms if m.janitor_passed) / len(ms)
            for role, ms in role_groups.items()
        }

        role_model_groups: dict[tuple[str, str], list[TaskMetrics]] = {}
        for m in task_metrics:
            key = (m.role or "unknown", m.model or "unknown")
            role_model_groups.setdefault(key, []).append(m)

        model_rates: dict[str, float] = {}
        recommendations: list[ModelRouteRecommendation] = []

        for (role, model), ms in sorted(role_model_groups.items()):
            if len(ms) < self._MIN_SAMPLES_PER_GROUP:
                continue
            rate = sum(1 for m in ms if m.janitor_passed) / len(ms)
            model_rates[f"{role}:{model}"] = rate
            failures = sum(1 for m in ms if not m.janitor_passed)

            if rate < self._AVOID_THRESHOLD:
                recommendations.append(
                    ModelRouteRecommendation(
                        role=role,
                        model=model,
                        success_rate=rate,
                        failure_count=failures,
                        recommendation="avoid",
                        reason=(
                            f"{role} tasks using {model} fail "
                            f"{1 - rate:.1%} of the time "
                            f"({failures}/{len(ms)} failures in {hours}h)"
                        ),
                    )
                )
            elif rate >= self._PREFER_THRESHOLD:
                recommendations.append(
                    ModelRouteRecommendation(
                        role=role,
                        model=model,
                        success_rate=rate,
                        failure_count=failures,
                        recommendation="prefer",
                        reason=(
                            f"{role} tasks using {model} succeed "
                            f"{rate:.1%} of the time in {hours}h"
                        ),
                    )
                )

        return SuccessRateAnalysis(
            overall_rate=overall,
            total_tasks=len(task_metrics),
            role_rates=role_rates,
            model_rates=model_rates,
            routing_recommendations=recommendations,
        )

    def export_routing_hints(self, output_path: Path) -> None:
        """Write routing recommendations to a JSON file for the orchestrator.

        The orchestrator reads this file on each tick to bias model selection
        toward high-success combinations and away from low-success ones.
        A missing or stale file is silently ignored by the orchestrator.

        Args:
            output_path: Destination path for the JSON output.
        """
        analysis = self.analyze()
        data: dict[str, Any] = {
            "generated_at": time.time(),
            "overall_success_rate": analysis.overall_rate,
            "total_tasks_analyzed": analysis.total_tasks,
            "role_success_rates": analysis.role_rates,
            "routing_recommendations": [
                {
                    "role": r.role,
                    "model": r.model,
                    "success_rate": r.success_rate,
                    "failure_count": r.failure_count,
                    "recommendation": r.recommendation,
                    "reason": r.reason,
                }
                for r in analysis.routing_recommendations
            ],
        }
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            logger.exception("Failed to write routing hints to %s", output_path)


class OpportunityDetector:
    """Identifies improvement opportunities from metrics."""

    def __init__(
        self,
        collector: MetricsCollector,
        failure_analyzer: FailureAnalyzer | None = None,
        analysis_dir: Path | None = None,
    ) -> None:
        self.collector = collector
        self.failure_analyzer = failure_analyzer
        self._analysis_dir = analysis_dir
        self._success_rate_advisor = SuccessRateAdvisor(collector)

    def identify_opportunities(self) -> list[ImprovementOpportunity]:
        """Identify improvement opportunities from recent metrics."""
        opportunities: list[ImprovementOpportunity] = []

        # Check for cost optimization opportunities
        cost_metrics = self.collector.get_recent_cost_metrics(hours=24)

        paid_providers = [m for m in cost_metrics if m.tier != "free"]
        if paid_providers:
            total_paid_cost = sum(m.cost_usd for m in paid_providers)
            if total_paid_cost > 1.0:  # More than $1 spent
                opportunities.append(
                    ImprovementOpportunity(
                        category=UpgradeCategory.ROUTING_RULES,
                        title="Optimize free tier utilization",
                        description="Consider routing more tasks to free tier providers",
                        expected_improvement=f"Potential savings of ${total_paid_cost * 0.3:.2f}/day",
                        confidence=0.7,
                        risk_level="low",
                        estimated_cost_impact_usd=-total_paid_cost * 0.3,
                    )
                )

        # Check for success rate improvements using the advisor for specificity
        analysis = self._success_rate_advisor.analyze(hours=24)
        if analysis.total_tasks > 0 and analysis.overall_rate < 0.8:
            avoid_recs = [r for r in analysis.routing_recommendations if r.recommendation == "avoid"]
            if avoid_recs:
                # Emit a targeted opportunity per problematic role/model pair
                for rec in avoid_recs:
                    opportunities.append(
                        ImprovementOpportunity(
                            category=UpgradeCategory.MODEL_ROUTING,
                            title="Improve task success rate",
                            description=(
                                f"Current success rate is {analysis.overall_rate:.1%}, target is 80%. "
                                f"{rec.reason}"
                            ),
                            expected_improvement=(
                                f"Routing {rec.role} tasks away from {rec.model} "
                                f"should reduce failure rate by up to "
                                f"{(0.8 - analysis.overall_rate):.1%}"
                            ),
                            confidence=min(0.9, 0.5 + rec.failure_count * 0.05),
                            risk_level="medium",
                            affected_components=["model_routing", rec.role],
                        )
                    )
            else:
                # No specific routing culprit — emit the generic proposal
                opportunities.append(
                    ImprovementOpportunity(
                        category=UpgradeCategory.MODEL_ROUTING,
                        title="Improve task success rate",
                        description=(
                            f"Current success rate is {analysis.overall_rate:.1%}, target is 80%"
                        ),
                        expected_improvement="Higher quality output, fewer fix tasks",
                        confidence=0.8,
                        risk_level="medium",
                        affected_components=["model_routing", "task_verification"],
                    )
                )

        # Check for failure-driven opportunities
        opportunities.extend(self.identify_failure_opportunities())

        if self._analysis_dir is not None:
            self._write_opportunities(opportunities)

        return opportunities

    def _write_opportunities(self, opportunities: list[ImprovementOpportunity]) -> None:
        """Write detected opportunities to .sdd/analysis/opportunities.json."""
        if self._analysis_dir is None:
            return
        try:
            self._analysis_dir.mkdir(parents=True, exist_ok=True)
            opportunities_path = self._analysis_dir / "opportunities.json"
            data = {
                "generated_at": time.time(),
                "count": len(opportunities),
                "opportunities": [asdict(o) for o in opportunities],
            }
            opportunities_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            logger.exception("Failed to write opportunities to %s", self._analysis_dir)

    def identify_failure_opportunities(self) -> list[ImprovementOpportunity]:
        """Identify improvement opportunities from recurring failure patterns.

        Analyzes detected failure patterns and generates targeted suggestions:
        - Single-model failures -> MODEL_ROUTING change
        - Single-role failures -> ROLE_TEMPLATES change
        - Broad failures -> POLICY_UPDATE

        Returns:
            List of improvement opportunities derived from failure analysis.
        """
        if self.failure_analyzer is None:
            return []

        patterns = self.failure_analyzer.detect_patterns()
        opportunities: list[ImprovementOpportunity] = []

        for pattern in patterns:
            if pattern.occurrence_count < 3:
                continue

            if len(pattern.affected_models) == 1:
                # Failures concentrated on a single model - route away from it
                model = pattern.affected_models[0]
                opportunities.append(
                    ImprovementOpportunity(
                        category=UpgradeCategory.MODEL_ROUTING,
                        title=f"Route {pattern.task_type} tasks away from {model}",
                        description=(
                            f"'{pattern.error_pattern}' occurred {pattern.occurrence_count} "
                            f"times on {model} for role {pattern.task_type}"
                        ),
                        expected_improvement=f"Reduce {pattern.task_type} failures by routing to alternative models",
                        confidence=min(0.9, 0.5 + pattern.occurrence_count * 0.05),
                        risk_level="medium",
                        affected_components=["model_routing"],
                    )
                )
            elif pattern.task_type and len(pattern.affected_models) != 1:
                # Failures spread across models but tied to a role - fix the template
                if len(pattern.affected_models) <= 1:
                    # Should not happen given outer condition, but defensive
                    category = UpgradeCategory.POLICY_UPDATE
                else:
                    category = UpgradeCategory.ROLE_TEMPLATES
                opportunities.append(
                    ImprovementOpportunity(
                        category=category,
                        title=f"Update template for {pattern.task_type} role",
                        description=(
                            f"'{pattern.error_pattern}' occurred {pattern.occurrence_count} "
                            f"times across models {', '.join(pattern.affected_models)} "
                            f"for role {pattern.task_type}"
                        ),
                        expected_improvement=f"Reduce recurring '{pattern.error_pattern}' failures",
                        confidence=min(0.9, 0.5 + pattern.occurrence_count * 0.05),
                        risk_level="medium",
                        affected_components=["role_templates", pattern.task_type],
                    )
                )
            else:
                # Broad pattern - suggest policy review
                opportunities.append(
                    ImprovementOpportunity(
                        category=UpgradeCategory.POLICY_UPDATE,
                        title=f"Review policy for '{pattern.error_pattern}' failures",
                        description=(
                            f"'{pattern.error_pattern}' occurred {pattern.occurrence_count} "
                            f"times across multiple roles and models"
                        ),
                        expected_improvement="Reduce systemic failure rate",
                        confidence=min(0.85, 0.4 + pattern.occurrence_count * 0.05),
                        risk_level="high",
                        affected_components=["policy"],
                    )
                )

        return opportunities


# ---------------------------------------------------------------------------
# Feature Discovery
# ---------------------------------------------------------------------------

# Keywords indicating retry logic is already present.
_RETRY_KEYWORDS: tuple[str, ...] = ("tenacity", "retry", "backoff")
# Keywords indicating caching is already present.
_CACHE_KEYWORDS: tuple[str, ...] = ("lru_cache", "functools.cache", "cache")
# Keywords indicating rate limiting is already present.
_RATE_KEYWORDS: tuple[str, ...] = ("ratelimit", "rate_limit", "throttle", "limits")

# Stop words excluded from keyword-overlap deduplication.
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "add",
        "the",
        "a",
        "an",
        "for",
        "to",
        "in",
        "of",
        "and",
        "or",
        "with",
        "use",
        "using",
        "improve",
        "implement",
        "create",
    }
)


@dataclass
class FeatureTicket:
    """A discovered feature opportunity written to the backlog."""

    title: str
    description: str
    role: str = "backend"
    priority: int = 2
    scope: str = "small"
    complexity: str = "medium"
    source: str = "todo_fixme"  # "todo_fixme" | "missing_pattern"
    ticket_id: str = ""
    file_path: Path | None = None


class FeatureDiscovery:
    """Discovers feature opportunities from codebase analysis.

    Scans ``src/`` for TODO/FIXME comments and detects common missing
    patterns (retry logic, caching, rate limiting).  Deduplicates against
    existing open/closed backlog tickets and caps output to avoid backlog
    bloat.

    Args:
        repo_root: Repository root directory.
        backlog_dir: Path to ``.sdd/backlog/``.
    """

    def __init__(self, repo_root: Path, backlog_dir: Path) -> None:
        self._repo_root = repo_root
        self._backlog_dir = backlog_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover(self, max_tickets: int = 5) -> list[FeatureTicket]:
        """Discover feature opportunities and write tickets to backlog/open/.

        Args:
            max_tickets: Maximum number of new tickets to generate per call.

        Returns:
            List of written FeatureTickets with ``ticket_id`` and
            ``file_path`` populated.
        """
        src_dir = self._repo_root / "src"
        if not src_dir.is_dir():
            return []

        existing_titles = self._load_existing_titles()
        candidates: list[FeatureTicket] = []
        candidates.extend(self._scan_todos(src_dir))
        candidates.extend(self._detect_missing_patterns(src_dir))

        # Deduplicate candidates against existing backlog and each other.
        seen: set[str] = set()
        filtered: list[FeatureTicket] = []
        for ticket in candidates:
            norm = _normalize(ticket.title)
            if _is_duplicate(norm, existing_titles):
                continue
            if norm in seen:
                continue
            seen.add(norm)
            filtered.append(ticket)

        selected = filtered[:max_tickets]

        next_id = self._next_ticket_id()
        result: list[FeatureTicket] = []
        for ticket in selected:
            ticket.ticket_id = str(next_id)
            ticket.file_path = self._write_ticket(ticket)
            result.append(ticket)
            next_id += 1

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_existing_titles(self) -> set[str]:
        """Return normalized titles from all open and closed backlog tickets."""
        titles: set[str] = set()
        _heading = re.compile(r"^#\s+\d+\s+-\s+(.+)$", re.MULTILINE)
        for subdir in ("open", "closed"):
            d = self._backlog_dir / subdir
            if not d.is_dir():
                continue
            for f in d.glob("*.md"):
                try:
                    content = f.read_text(encoding="utf-8")
                except OSError:
                    continue
                m = _heading.search(content)
                if m:
                    titles.add(_normalize(m.group(1).strip()))
        return titles

    def _scan_todos(self, src_dir: Path) -> list[FeatureTicket]:
        """Scan ``src/`` for TODO/FIXME comments and return candidate tickets."""
        pattern = re.compile(r"#\s*(?:TODO|FIXME)[:\s]+(.+)", re.IGNORECASE)
        tickets: list[FeatureTicket] = []
        for py_file in sorted(src_dir.rglob("*.py")):
            # Skip test files that happen to live under src/.
            if any(part.startswith("test") for part in py_file.relative_to(src_dir).parts):
                continue
            try:
                content = py_file.read_text(encoding="utf-8")
            except OSError:
                continue
            for m in pattern.finditer(content):
                text = m.group(1).strip()
                if not text:
                    continue
                title = text[0].upper() + text[1:]
                tickets.append(
                    FeatureTicket(
                        title=title,
                        description=(f"Found in {py_file.relative_to(self._repo_root)}: # {m.group(0).strip()}"),
                        source="todo_fixme",
                    )
                )
        return tickets

    def _detect_missing_patterns(self, src_dir: Path) -> list[FeatureTicket]:
        """Return tickets for common patterns not yet present in src/."""
        all_content = ""
        for py_file in src_dir.rglob("*.py"):
            with contextlib.suppress(OSError):
                all_content += py_file.read_text(encoding="utf-8").lower()

        tickets: list[FeatureTicket] = []

        if not any(kw in all_content for kw in _RETRY_KEYWORDS):
            tickets.append(
                FeatureTicket(
                    title="Add retry logic for transient failures",
                    description=("No retry logic detected in src/. Add tenacity (or equivalent) to harden API calls."),
                    source="missing_pattern",
                )
            )

        if not any(kw in all_content for kw in _CACHE_KEYWORDS):
            tickets.append(
                FeatureTicket(
                    title="Add caching for expensive operations",
                    description=(
                        "No caching pattern detected in src/. "
                        "Use functools.lru_cache or similar to avoid redundant work."
                    ),
                    source="missing_pattern",
                )
            )

        if not any(kw in all_content for kw in _RATE_KEYWORDS):
            tickets.append(
                FeatureTicket(
                    title="Add rate limiting for external API calls",
                    description=(
                        "No rate-limiting pattern detected in src/. Add rate limiting to prevent quota exhaustion."
                    ),
                    source="missing_pattern",
                )
            )

        return tickets

    def _next_ticket_id(self) -> int:
        """Return the next available numeric ticket ID."""
        max_id = 0
        for subdir in ("open", "closed"):
            d = self._backlog_dir / subdir
            if not d.is_dir():
                continue
            for f in d.glob("*.md"):
                m = re.match(r"(\d+)", f.name)
                if m:
                    max_id = max(max_id, int(m.group(1)))
        return max_id + 1

    def _write_ticket(self, ticket: FeatureTicket) -> Path:
        """Serialize a ticket to backlog/open/ and return its path."""
        open_dir = self._backlog_dir / "open"
        open_dir.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "-", ticket.title.lower()).strip("-")[:50]
        filename = f"{ticket.ticket_id}-{slug}.md"
        file_path = open_dir / filename
        content = (
            f"# {ticket.ticket_id} - {ticket.title}\n\n"
            f"**Role:** {ticket.role}\n"
            f"**Priority:** {ticket.priority}\n"
            f"**Scope:** {ticket.scope}\n"
            f"**Complexity:** {ticket.complexity}\n\n"
            f"## Description\n\n"
            f"{ticket.description}\n\n"
            f"<!-- source: feature-discovery -->\n"
        )
        file_path.write_text(content, encoding="utf-8")
        return file_path


def _normalize(text: str) -> str:
    """Lowercase and strip punctuation for fuzzy comparison."""
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()


def _is_duplicate(normalized: str, existing: set[str]) -> bool:
    """Return True if *normalized* is semantically similar to any existing title."""
    words = set(normalized.split()) - _STOP_WORDS
    if not words:
        return False
    for existing_title in existing:
        existing_words = set(existing_title.split()) - _STOP_WORDS
        if not existing_words:
            continue
        overlap = len(words & existing_words) / min(len(words), len(existing_words))
        if overlap >= 0.6:
            return True
    return False
