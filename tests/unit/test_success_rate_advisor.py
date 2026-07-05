"""Tests for SuccessRateAdvisor and updated OpportunityDetector success-rate path."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from bernstein.evolution.aggregator import FileMetricsCollector, TaskMetrics
from bernstein.evolution.detector import (
    OpportunityDetector,
    SuccessRateAdvisor,
    UpgradeCategory,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metric(
    *,
    role: str = "backend",
    model: str = "sonnet",
    janitor_passed: bool = True,
    cost_usd: float = 0.01,
) -> TaskMetrics:
    return TaskMetrics(
        timestamp=time.time(),
        task_id=f"T-{id(object())}",
        role=role,
        model=model,
        cost_usd=cost_usd,
        janitor_passed=janitor_passed,
    )


def _seed(collector: FileMetricsCollector, metrics: list[TaskMetrics]) -> None:
    for m in metrics:
        collector.record_task_metrics(m)


# ---------------------------------------------------------------------------
# SuccessRateAdvisor - empty data
# ---------------------------------------------------------------------------


def test_advisor_empty_returns_zero(tmp_path: Path) -> None:
    """No metrics → overall_rate=0, empty dicts and recommendations."""
    collector = FileMetricsCollector(tmp_path)
    advisor = SuccessRateAdvisor(collector)
    result = advisor.analyze()

    assert result.overall_rate == 0.0
    assert result.total_tasks == 0
    assert result.role_rates == {}
    assert result.model_rates == {}
    assert result.routing_recommendations == []


# ---------------------------------------------------------------------------
# SuccessRateAdvisor - overall rate
# ---------------------------------------------------------------------------


def test_advisor_overall_rate_all_pass(tmp_path: Path) -> None:
    collector = FileMetricsCollector(tmp_path)
    _seed(collector, [_make_metric(janitor_passed=True) for _ in range(5)])
    result = SuccessRateAdvisor(collector).analyze()
    assert result.overall_rate == pytest.approx(1.0)
    assert result.total_tasks == 5


def test_advisor_overall_rate_mixed(tmp_path: Path) -> None:
    collector = FileMetricsCollector(tmp_path)
    _seed(
        collector,
        [_make_metric(janitor_passed=True)] * 3 + [_make_metric(janitor_passed=False)],
    )
    result = SuccessRateAdvisor(collector).analyze()
    assert result.overall_rate == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# SuccessRateAdvisor - role rates
# ---------------------------------------------------------------------------


def test_advisor_role_rates(tmp_path: Path) -> None:
    collector = FileMetricsCollector(tmp_path)
    _seed(
        collector,
        [
            _make_metric(role="backend", janitor_passed=True),
            _make_metric(role="backend", janitor_passed=False),
            _make_metric(role="qa", janitor_passed=True),
        ],
    )
    result = SuccessRateAdvisor(collector).analyze()
    assert result.role_rates["backend"] == pytest.approx(0.5)
    assert result.role_rates["qa"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# SuccessRateAdvisor - routing recommendations
# ---------------------------------------------------------------------------


def test_advisor_avoid_recommendation(tmp_path: Path) -> None:
    """A (role, model) pair with <50% success rate produces an 'avoid' recommendation."""
    collector = FileMetricsCollector(tmp_path)
    # 1 pass, 4 fail → 20% success → below AVOID_THRESHOLD
    _seed(
        collector,
        [_make_metric(role="backend", model="haiku", janitor_passed=True)]
        + [_make_metric(role="backend", model="haiku", janitor_passed=False)] * 4,
    )
    result = SuccessRateAdvisor(collector).analyze()
    recs = result.routing_recommendations
    assert len(recs) == 1
    assert recs[0].recommendation == "avoid"
    assert recs[0].role == "backend"
    assert recs[0].model == "haiku"
    assert recs[0].failure_count == 4
    assert recs[0].success_rate == pytest.approx(0.2)


def test_advisor_prefer_recommendation(tmp_path: Path) -> None:
    """A (role, model) pair with >=90% success rate produces a 'prefer' recommendation."""
    collector = FileMetricsCollector(tmp_path)
    # 9 pass, 1 fail → 90% success → at PREFER_THRESHOLD
    _seed(
        collector,
        [_make_metric(role="backend", model="opus", janitor_passed=True)] * 9
        + [_make_metric(role="backend", model="opus", janitor_passed=False)],
    )
    result = SuccessRateAdvisor(collector).analyze()
    recs = result.routing_recommendations
    assert len(recs) == 1
    assert recs[0].recommendation == "prefer"
    assert recs[0].model == "opus"


def test_advisor_below_min_samples_no_recommendation(tmp_path: Path) -> None:
    """Fewer than _MIN_SAMPLES_PER_GROUP (3) samples → no recommendation."""
    collector = FileMetricsCollector(tmp_path)
    _seed(
        collector,
        [_make_metric(role="backend", model="haiku", janitor_passed=False)] * 2,
    )
    result = SuccessRateAdvisor(collector).analyze()
    assert result.routing_recommendations == []


def test_advisor_middle_band_no_recommendation(tmp_path: Path) -> None:
    """50-89% success rate → neither avoid nor prefer."""
    collector = FileMetricsCollector(tmp_path)
    # 50% success: exactly at boundary
    _seed(
        collector,
        [_make_metric(role="backend", model="sonnet", janitor_passed=True)] * 3
        + [_make_metric(role="backend", model="sonnet", janitor_passed=False)] * 3,
    )
    result = SuccessRateAdvisor(collector).analyze()
    assert result.routing_recommendations == []


# ---------------------------------------------------------------------------
# SuccessRateAdvisor - export_routing_hints
# ---------------------------------------------------------------------------


def test_advisor_export_routing_hints(tmp_path: Path) -> None:
    """export_routing_hints writes a JSON file with expected keys."""
    collector = FileMetricsCollector(tmp_path)
    _seed(
        collector,
        [_make_metric(role="backend", model="haiku", janitor_passed=False)] * 5,
    )
    output = tmp_path / "hints" / "routing_hints.json"
    SuccessRateAdvisor(collector).export_routing_hints(output)

    assert output.exists()
    data = json.loads(output.read_text())
    assert "generated_at" in data
    assert "overall_success_rate" in data
    assert "total_tasks_analyzed" in data
    assert "routing_recommendations" in data
    assert isinstance(data["routing_recommendations"], list)


def test_advisor_export_creates_parent_dir(tmp_path: Path) -> None:
    """export_routing_hints creates missing parent directories."""
    collector = FileMetricsCollector(tmp_path)
    _seed(collector, [_make_metric()])
    nested = tmp_path / "a" / "b" / "c" / "hints.json"
    SuccessRateAdvisor(collector).export_routing_hints(nested)
    assert nested.exists()


# ---------------------------------------------------------------------------
# OpportunityDetector - success rate with advisor
# ---------------------------------------------------------------------------


def test_opportunity_detector_specific_proposal_with_bad_model(tmp_path: Path) -> None:
    """When a model has <50% success, the proposal description names the model."""
    collector = FileMetricsCollector(tmp_path)
    # Seed: 5 failures from haiku, overall rate < 80%
    _seed(
        collector,
        [_make_metric(role="backend", model="haiku", janitor_passed=False)] * 5
        + [_make_metric(role="backend", model="haiku", janitor_passed=True)],
    )
    detector = OpportunityDetector(collector)
    opps = detector.identify_opportunities()

    success_opps = [o for o in opps if "success rate" in o.title.lower()]
    assert len(success_opps) >= 1
    # The description should name the problematic model
    descriptions = " ".join(o.description for o in success_opps)
    assert "haiku" in descriptions


def test_opportunity_detector_generic_proposal_no_routing_culprit(tmp_path: Path) -> None:
    """With no specific routing culprit, a generic proposal is emitted."""
    collector = FileMetricsCollector(tmp_path)
    # 30% pass rate but only 2 samples per model (below min_samples), so no
    # model-specific recommendations → falls back to generic proposal
    _seed(
        collector,
        [
            _make_metric(role="backend", model="sonnet", janitor_passed=False),
            _make_metric(role="backend", model="haiku", janitor_passed=False),
            _make_metric(role="backend", model="sonnet", janitor_passed=True),
        ],
    )
    detector = OpportunityDetector(collector)
    opps = detector.identify_opportunities()

    success_opps = [o for o in opps if "success rate" in o.title.lower()]
    assert len(success_opps) >= 1
    assert success_opps[0].category == UpgradeCategory.MODEL_ROUTING


def test_opportunity_detector_no_proposal_when_rate_above_threshold(tmp_path: Path) -> None:
    """No success-rate proposal when pass_rate >= 80%."""
    collector = FileMetricsCollector(tmp_path)
    # 9/10 = 90% pass rate
    _seed(
        collector,
        [_make_metric(janitor_passed=True)] * 9 + [_make_metric(janitor_passed=False)],
    )
    detector = OpportunityDetector(collector)
    opps = detector.identify_opportunities()
    success_opps = [o for o in opps if "success rate" in o.title.lower()]
    assert success_opps == []
