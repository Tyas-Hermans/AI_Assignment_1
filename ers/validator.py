"""Independent validation of a plan against a scenario.

This module deliberately does **not** trust the planner's internal state. Given
only a scenario folder and a ``plan.json``-shaped list, it re-derives the road
beliefs and the blocked set, then checks every operational constraint from the
assignment:

- the path starts at the responder's current location and ends at the incident;
- every step of the path is an existing road;
- no road believed blocked is used;
- the responder has the required capability;
- the start time is ≥ the incident timestamp and ≥ the responder's free time;
- a responder never has two overlapping assignments;
- each incident is assigned at most once;
- whether the deadline is met (reported, not required).

Limitation: arrival times (and therefore deadline checks) are recomputed from
the same *expected* cost model the planner uses, so the validator confirms
internal consistency and constraint satisfaction, not what would happen under a
specific real road outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .bayes import compute_road_status_beliefs
from .data_loader import Scenario, load_scenario
from .graph import PlanningGraph, PlanningParams, build_planning_graph
from .hmm import compute_traffic_beliefs


@dataclass
class AssignmentCheck:
    incident_id: str
    responder_id: str
    errors: List[str] = field(default_factory=list)
    arrival_time: Optional[float] = None
    meets_deadline: Optional[bool] = None

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class ValidationReport:
    checks: List[AssignmentCheck]
    global_errors: List[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.global_errors and all(c.ok for c in self.checks)

    def summary(self) -> str:
        lines = [f"Plan validation: {'VALID' if self.valid else 'INVALID'}"]
        for err in self.global_errors:
            lines.append(f"  [global] {err}")
        for c in self.checks:
            status = "ok" if c.ok else "FAIL"
            deadline = (
                "n/a"
                if c.meets_deadline is None
                else ("met" if c.meets_deadline else "MISSED")
            )
            lines.append(
                f"  [{status}] {c.responder_id} -> {c.incident_id} "
                f"(deadline {deadline})"
            )
            for err in c.errors:
                lines.append(f"      - {err}")
        return "\n".join(lines)


def build_graph_for(scenario: Scenario, params: PlanningParams = PlanningParams()) -> PlanningGraph:
    status = compute_road_status_beliefs(scenario)
    traffic = compute_traffic_beliefs(scenario)
    return build_planning_graph(scenario, status, traffic, params)


def validate_plan(
    scenario: Scenario,
    plan: List[dict],
    params: PlanningParams = PlanningParams(),
) -> ValidationReport:
    """Validate a plan (list of assignment dicts) against a scenario."""
    graph = build_graph_for(scenario, params)
    responders = {r.id: r for r in scenario.responders}
    incidents = {i.id: i for i in scenario.incidents}

    report = ValidationReport(checks=[])

    # Per-responder running state, to check availability and overlaps in time.
    current_location: Dict[str, str] = {r.id: r.start_location for r in scenario.responders}
    free_time: Dict[str, float] = {r.id: r.free_at for r in scenario.responders}
    seen_incidents: set = set()

    # Process assignments in dispatch-time order, as they would actually occur.
    for entry in sorted(plan, key=lambda e: (e.get("time", 0.0), e.get("incident_id", ""))):
        rid = entry.get("responder_id")
        iid = entry.get("incident_id")
        path = entry.get("path", [])
        start = float(entry.get("time", 0.0))
        check = AssignmentCheck(incident_id=iid, responder_id=rid)

        if rid not in responders:
            check.errors.append(f"unknown responder '{rid}'")
            report.checks.append(check)
            continue
        if iid not in incidents:
            check.errors.append(f"unknown incident '{iid}'")
            report.checks.append(check)
            continue

        responder = responders[rid]
        incident = incidents[iid]

        # Each incident at most once.
        if iid in seen_incidents:
            check.errors.append("incident assigned more than once")
        seen_incidents.add(iid)

        # Capability match.
        if incident.required_capability not in responder.capabilities:
            check.errors.append(
                f"missing capability '{incident.required_capability}'"
            )

        # Timing: not before the incident is known, not before the responder is free.
        if start < incident.timestamp - 1e-9:
            check.errors.append(
                f"starts at {start} before incident timestamp {incident.timestamp}"
            )
        if start < free_time[rid] - 1e-9:
            check.errors.append(
                f"starts at {start} before responder free at {free_time[rid]}"
            )

        # Path endpoints.
        if not path:
            check.errors.append("empty path")
        else:
            if path[0] != current_location[rid]:
                check.errors.append(
                    f"path starts at {path[0]}, responder is at {current_location[rid]}"
                )
            if path[-1] != incident.location:
                check.errors.append(
                    f"path ends at {path[-1]}, incident is at {incident.location}"
                )

        # Every step is a real, non-blocked road; accumulate travel time.
        travel = 0.0
        routable = True
        for u, v in zip(path, path[1:]):
            key = (u, v)
            if key not in scenario.graph.roads:
                check.errors.append(f"road {u}->{v} does not exist")
                routable = False
                continue
            if key in graph.blocked:
                check.errors.append(f"road {u}->{v} is believed blocked")
            travel += graph.costs.get(key, 0.0)

        if routable and path and not check.errors:
            arrival = start + travel
            check.arrival_time = arrival
            check.meets_deadline = arrival <= incident.deadline + 1e-9
            # Advance this responder's state.
            free_time[rid] = arrival + incident.time_required
            current_location[rid] = incident.location
        elif path and path[-1] == incident.location:
            # Still report a deadline verdict when we could compute travel.
            arrival = start + travel
            check.arrival_time = arrival
            check.meets_deadline = arrival <= incident.deadline + 1e-9

        report.checks.append(check)

    return report


def validate_plan_file(scenario_folder: str, plan_path: str) -> ValidationReport:
    """Convenience wrapper: load a scenario and a plan.json file, then validate."""
    import json

    scenario = load_scenario(scenario_folder)
    with open(plan_path, "r", encoding="utf-8") as fh:
        plan = json.load(fh)
    return validate_plan(scenario, plan)
