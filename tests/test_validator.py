"""Tests for the independent plan validator.

These check both directions: a correct plan passes, and each kind of corrupted
plan is actually rejected (so the validator is not a rubber stamp).
"""

import copy

import pytest

from ers.planner import plan_scenario
from ers.validator import validate_plan


def planned(scenario_folder):
    result = plan_scenario(scenario_folder)
    return result.scenario, result.to_plan_json()


def test_generated_plan_is_valid(any_scenario_folder):
    scenario, plan = planned(any_scenario_folder)
    report = validate_plan(scenario, plan)
    assert report.valid, report.summary()


def test_rejects_capability_mismatch(scenario1_folder):
    scenario, plan = planned(scenario1_folder)
    # Swap in a responder that lacks the required capability.
    bad = copy.deepcopy(plan)
    bad[0]["responder_id"] = _responder_without_capability(scenario, bad[0])
    report = validate_plan(scenario, bad)
    assert not report.valid


def test_rejects_nonexistent_road(scenario1_folder):
    scenario, plan = planned(scenario1_folder)
    bad = copy.deepcopy(plan)
    # Insert a fake node into the path so a step is not a real road.
    bad[0]["path"] = [bad[0]["path"][0], "GHOST", bad[0]["path"][-1]]
    report = validate_plan(scenario, bad)
    assert not report.valid
    assert any("does not exist" in e for c in report.checks for e in c.errors)


def test_rejects_blocked_road(scenario1_folder):
    scenario, plan = planned(scenario1_folder)
    # (N1,N2) is believed blocked in Scenario1; force a plan step to use it.
    bad = copy.deepcopy(plan)
    bad.append(
        {"responder_id": plan[0]["responder_id"], "incident_id": plan[0]["incident_id"],
         "path": ["N1", "N2"], "time": 0}
    )
    report = validate_plan(scenario, bad)
    assert not report.valid
    assert any("blocked" in e for c in report.checks for e in c.errors)


def test_rejects_duplicate_incident(scenario1_folder):
    scenario, plan = planned(scenario1_folder)
    bad = copy.deepcopy(plan)
    bad.append(copy.deepcopy(bad[0]))  # same incident twice
    report = validate_plan(scenario, bad)
    assert not report.valid
    assert any("more than once" in e for c in report.checks for e in c.errors)


def test_rejects_start_before_timestamp(scenario1_folder):
    scenario, plan = planned(scenario1_folder)
    # Find an incident with a positive timestamp and dispatch it too early.
    late = next(i for i in scenario.incidents if i.timestamp > 0)
    bad = copy.deepcopy(plan)
    bad.append(
        {"responder_id": plan[0]["responder_id"], "incident_id": late.id,
         "path": [scenario.responders[0].start_location], "time": -1}
    )
    report = validate_plan(scenario, bad)
    assert not report.valid


def _responder_without_capability(scenario, entry):
    incident = next(i for i in scenario.incidents if i.id == entry["incident_id"])
    for r in scenario.responders:
        if incident.required_capability not in r.capabilities:
            return r.id
    pytest.skip("no capability-mismatched responder available")
