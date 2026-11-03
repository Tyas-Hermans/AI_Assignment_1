"""Unit tests for the event-driven CSP allocation."""

import pytest

from ers.allocation import _RouteCache, run_simulation, solve_event_csp
from ers.bayes import compute_road_status_beliefs
from ers.data_loader import Incident, Node, Responder
from ers.graph import PlanningGraph, build_planning_graph
from ers.hmm import compute_traffic_beliefs


def make_graph(scenario):
    status = compute_road_status_beliefs(scenario)
    traffic = compute_traffic_beliefs(scenario)
    return build_planning_graph(scenario, status, traffic)


# --------------------------------------------------------------------------- #
# Invariants on the real scenarios
# --------------------------------------------------------------------------- #

def test_assignments_respect_all_constraints(any_scenario):
    graph = make_graph(any_scenario)
    result = run_simulation(any_scenario, graph)
    responders = {r.id: r for r in any_scenario.responders}
    incidents = {i.id: i for i in any_scenario.incidents}

    seen_incidents = set()
    busy_intervals = {}
    for a in result.assignments:
        incident = incidents[a.incident_id]
        responder = responders[a.responder_id]

        # Capability match.
        assert incident.required_capability in responder.capabilities
        # Cannot start before the incident is known or before the responder is free.
        assert a.start_time >= incident.timestamp - 1e-9
        assert a.start_time >= responder.free_at - 1e-9
        # Path starts at the responder's location at dispatch and ends at incident.
        assert a.path[-1] == incident.location
        # Each incident assigned at most once.
        assert a.incident_id not in seen_incidents
        seen_incidents.add(a.incident_id)

        # No overlapping jobs per responder.
        interval = (a.start_time, a.arrival_time + incident.time_required)
        for prev in busy_intervals.get(a.responder_id, []):
            assert interval[0] >= prev[1] - 1e-9 or interval[1] <= prev[0] + 1e-9
        busy_intervals.setdefault(a.responder_id, []).append(interval)

    # Handled + unhandled partition the incident set exactly.
    assert seen_incidents.isdisjoint(result.unhandled_incident_ids)
    assert seen_incidents | set(result.unhandled_incident_ids) == set(incidents)


def test_reachable_incidents_are_never_skipped(any_scenario):
    """Serve-late policy: an incident some capable responder can reach is served,
    even if every option is late. Only unreachable incidents may stay unhandled."""
    graph = make_graph(any_scenario)
    result = run_simulation(any_scenario, graph)
    routes = _RouteCache(graph)
    incidents = {i.id: i for i in any_scenario.incidents}
    for iid in result.unhandled_incident_ids:
        inc = incidents[iid]
        # No capable responder may be able to reach it from its start location.
        reachable = [
            r.id
            for r in any_scenario.responders
            if inc.required_capability in r.capabilities
            and routes.route(r.start_location, inc.location) is not None
        ]
        assert reachable == [], f"{iid} reachable by {reachable} but left unhandled"
    # Every assignment records lateness consistently with its deadline.
    for a in result.assignments:
        assert a.meets_deadline == (a.arrival_time <= a.deadline + 1e-9)


# --------------------------------------------------------------------------- #
# Synthetic CSP test: backtracking + MRV must reshuffle where greedy fails
# --------------------------------------------------------------------------- #

def _tiny_graph():
    """Four co-located nodes with hand-set directed costs.

    Reachability:
      R1@P1 -> A (cost 1, cheapest for A) and -> B (cost 5)
      R2@P2 -> A (cost 3) only;  R2 cannot reach B.
    So B is only servable by R1; A is servable by R1 or R2.
    """
    nodes = {n: Node(n, n, 0.0, 0.0, "normal") for n in ("P1", "P2", "A", "B")}
    adjacency = {
        "P1": [("A", 1.0, ("P1", "A")), ("B", 5.0, ("P1", "B"))],
        "P2": [("A", 3.0, ("P2", "A"))],
        "A": [],
        "B": [],
    }
    costs = {("P1", "A"): 1.0, ("P1", "B"): 5.0, ("P2", "A"): 3.0}
    return PlanningGraph(nodes, adjacency, costs, set(), max_speed=1.0)


def test_backtracking_prefers_serving_critical_and_reshuffles():
    graph = _tiny_graph()
    routes = _RouteCache(graph)

    inc_a = Incident("A", "A", "High", 100, "cap", 0, 0)       # servable by R1 or R2
    inc_b = Incident("B", "B", "Critical", 100, "cap", 0, 0)   # servable by R1 only
    r1 = Responder("R1", "t", "P1", frozenset({"cap"}), "Available", 0.0)
    r2 = Responder("R2", "t", "P2", frozenset({"cap"}), "Available", 0.0)
    responders = {"R1": r1, "R2": r2}
    locations = {"R1": "P1", "R2": "P2"}

    choice = solve_event_csp(
        [inc_a, inc_b], ["R1", "R2"], locations, responders, now=0.0, routes=routes
    )

    # A naive "give each incident its cheapest responder" would hand R1 to A and
    # strand the Critical B. MRV/backtracking must serve both, giving B its only
    # option (R1) and A the remaining R2.
    assert set(choice) == {"A", "B"}
    assert choice["B"].responder_id == "R1"
    assert choice["A"].responder_id == "R2"


def test_missed_deadline_is_served_late_not_skipped():
    graph = _tiny_graph()
    routes = _RouteCache(graph)
    # B only reachable by R1 at cost 5, but deadline is 4 → cannot be on time.
    # Serve-late policy: R1 is still dispatched, arriving 1 s late.
    inc_b = Incident("B", "B", "Critical", 4, "cap", 0, 0)
    r1 = Responder("R1", "t", "P1", frozenset({"cap"}), "Available", 0.0)
    choice = solve_event_csp(
        [inc_b], ["R1"], {"R1": "P1"}, {"R1": r1}, now=0.0, routes=routes
    )
    assert set(choice) == {"B"}
    assert choice["B"].responder_id == "R1"
    assert not choice["B"].on_time
    assert choice["B"].lateness == pytest.approx(1.0)


def test_on_time_responder_preferred_over_late_one():
    graph = _tiny_graph()
    routes = _RouteCache(graph)
    # A: R1 arrives at 1 (on time), R2 at 3 (late, deadline 2). Pick R1.
    inc_a = Incident("A", "A", "High", 2, "cap", 0, 0)
    r1 = Responder("R1", "t", "P1", frozenset({"cap"}), "Available", 0.0)
    r2 = Responder("R2", "t", "P2", frozenset({"cap"}), "Available", 0.0)
    choice = solve_event_csp(
        [inc_a], ["R1", "R2"], {"R1": "P1", "R2": "P2"},
        {"R1": r1, "R2": r2}, now=0.0, routes=routes,
    )
    assert choice["A"].responder_id == "R1"
    assert choice["A"].on_time


def test_capability_mismatch_leaves_incident_unassigned():
    graph = _tiny_graph()
    routes = _RouteCache(graph)
    inc_a = Incident("A", "A", "High", 100, "needs_other", 0, 0)
    r2 = Responder("R2", "t", "P2", frozenset({"cap"}), "Available", 0.0)
    choice = solve_event_csp(
        [inc_a], ["R2"], {"R2": "P2"}, {"R2": r2}, now=0.0, routes=routes
    )
    assert choice == {}
