"""Orchestrate the full pipeline for one scenario.

Wires the four AI components together:

    load → Bayes (status belief) + HMM (traffic belief)
         → graph (expected travel times) → allocation (event-driven CSP)

and exposes the result in the shape the output writers and validator need.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .allocation import Assignment, AllocationResult, run_simulation
from .bayes import compute_road_status_beliefs
from .data_loader import Scenario, load_scenario
from .graph import PlanningGraph, PlanningParams, build_planning_graph
from .hmm import compute_traffic_beliefs


@dataclass
class PlanResult:
    """The complete outcome of planning one scenario."""

    scenario: Scenario
    graph: PlanningGraph
    assignments: List[Assignment]
    unhandled_incident_ids: List[str]

    def to_plan_json(self) -> list:
        """Serialise assignments into the required plan.json structure.

        Each entry: responder_id, incident_id, path (node ids), and time (the
        departure time). Ordered by departure time for readability.
        """
        ordered = sorted(self.assignments, key=lambda a: (a.start_time, a.incident_id))
        return [
            {
                "responder_id": a.responder_id,
                "incident_id": a.incident_id,
                "path": a.path,
                "time": _round(a.start_time),
            }
            for a in ordered
        ]


def _round(value: float) -> float:
    """Return an int for whole values, else the exact float.

    We deliberately do NOT round non-integer departure times. plan.json stores
    only the departure time; the validator reconstructs arrival and the next
    departure from it. Rounding here would make a responder appear to leave a
    fraction of a second before it is actually free, breaking that
    reconstruction. Clean scenarios still yield integer times (e.g. 0, 10).
    """
    if value == int(value):
        return int(value)
    return value


def plan_scenario(
    folder: str, params: PlanningParams = PlanningParams()
) -> PlanResult:
    """Run the whole pipeline on a scenario folder and return the plan."""
    scenario = load_scenario(folder)
    status_beliefs = compute_road_status_beliefs(scenario)
    traffic_beliefs = compute_traffic_beliefs(scenario)
    graph = build_planning_graph(scenario, status_beliefs, traffic_beliefs, params)
    result: AllocationResult = run_simulation(scenario, graph)
    return PlanResult(
        scenario=scenario,
        graph=graph,
        assignments=result.assignments,
        unhandled_incident_ids=result.unhandled_incident_ids,
    )
