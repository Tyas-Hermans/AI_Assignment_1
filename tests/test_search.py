"""Unit tests for the planning graph and A* search."""

import pytest

from ers.bayes import compute_road_status_beliefs
from ers.graph import PlanningParams, build_planning_graph
from ers.hmm import compute_traffic_beliefs
from ers.search import a_star, dijkstra


def make_graph(scenario, **overrides):
    status = compute_road_status_beliefs(scenario)
    traffic = compute_traffic_beliefs(scenario)
    params = PlanningParams(**overrides) if overrides else PlanningParams()
    return build_planning_graph(scenario, status, traffic, params)


def test_heuristic_is_admissible(any_scenario):
    """euclidean/max_speed must never exceed the true optimal cost."""
    graph = make_graph(any_scenario)
    nodes = list(graph.nodes)
    for start in nodes:
        for goal in nodes:
            result = dijkstra(graph, start, goal)
            if result is None:
                continue
            _, true_cost = result
            assert graph.heuristic(start, goal) <= true_cost + 1e-9


def test_astar_matches_dijkstra_cost(any_scenario):
    """A* with an admissible heuristic returns the optimal (Dijkstra) cost."""
    graph = make_graph(any_scenario)
    nodes = list(graph.nodes)
    for start in nodes:
        for goal in nodes:
            a = a_star(graph, start, goal)
            d = dijkstra(graph, start, goal)
            assert (a is None) == (d is None)
            if a is not None:
                assert a[1] == pytest.approx(d[1], rel=1e-9, abs=1e-9)


def test_astar_path_is_connected_and_valid(scenario1):
    graph = make_graph(scenario1)
    result = a_star(graph, "N3", "N4")
    assert result is not None
    path, cost = result
    assert path[0] == "N3" and path[-1] == "N4"
    assert cost > 0
    # Every consecutive pair is a real, non-blocked road.
    for u, v in zip(path, path[1:]):
        assert (u, v) in graph.costs
        assert (u, v) not in graph.blocked


def test_same_node_route_is_zero(scenario1):
    graph = make_graph(scenario1)
    assert a_star(graph, "N1", "N1") == (["N1"], 0.0)


def test_blocked_road_removed_from_graph(scenario1):
    # (N1,N2) has P(Blocked) ~0.527 > 0.5, so it must be dropped.
    graph = make_graph(scenario1)
    assert ("N1", "N2") in graph.blocked
    assert ("N1", "N2") not in graph.costs
    # The dropped road must not appear as an outgoing edge from N1 either.
    assert "N2" not in [to for to, _cost, _key in graph.adjacency["N1"]]


def test_lower_blocked_threshold_drops_more_roads(scenario1):
    lenient = make_graph(scenario1, blocked_threshold=0.9)
    strict = make_graph(scenario1, blocked_threshold=0.1)
    assert len(strict.blocked) >= len(lenient.blocked)
