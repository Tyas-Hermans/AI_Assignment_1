"""Shortest-route search over the weighted planning graph.

Course concept: **informed search with A***, plus Dijkstra as a correctness
oracle.

``a_star`` finds the minimum expected-travel-time route between two nodes using
the straight-line/max-speed heuristic from :meth:`PlanningGraph.heuristic`.
Because that heuristic is admissible (and consistent), A* returns an
optimal-cost path — the same cost Dijkstra finds. We keep ``dijkstra`` in the
codebase purely so a unit test can assert the two agree.
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple

from .graph import PlanningGraph

# A route result: (ordered list of node ids, total expected time in seconds).
Route = Tuple[List[str], float]


def a_star(graph: PlanningGraph, start: str, goal: str) -> Optional[Route]:
    """Minimum-cost route from ``start`` to ``goal`` using A*.

    Returns ``(path, cost)`` where ``path`` begins at ``start`` and ends at
    ``goal``, or ``None`` if no route exists (e.g. the goal is cut off by
    blocked roads). A trivial path is returned when ``start == goal``.
    """
    if start == goal:
        return ([start], 0.0)

    # Best known cost-so-far (g) per node, and the node we came from.
    g_score: Dict[str, float] = {start: 0.0}
    came_from: Dict[str, str] = {}

    # Priority queue ordered by f = g + h. The counter breaks ties so heapq
    # never compares node strings (and keeps ordering deterministic).
    counter = 0
    open_heap: List[Tuple[float, int, str]] = [
        (graph.heuristic(start, goal), counter, start)
    ]
    closed: set = set()

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current == goal:
            return (_reconstruct(came_from, current), g_score[current])
        if current in closed:
            continue
        closed.add(current)

        for neighbour, cost, _key in graph.adjacency.get(current, []):
            tentative = g_score[current] + cost
            if tentative < g_score.get(neighbour, float("inf")):
                came_from[neighbour] = current
                g_score[neighbour] = tentative
                counter += 1
                f = tentative + graph.heuristic(neighbour, goal)
                heapq.heappush(open_heap, (f, counter, neighbour))

    return None


def dijkstra(graph: PlanningGraph, start: str, goal: str) -> Optional[Route]:
    """Uniform-cost search (A* with a zero heuristic). Test oracle only."""
    if start == goal:
        return ([start], 0.0)

    g_score: Dict[str, float] = {start: 0.0}
    came_from: Dict[str, str] = {}
    counter = 0
    open_heap: List[Tuple[float, int, str]] = [(0.0, counter, start)]
    closed: set = set()

    while open_heap:
        dist, _, current = heapq.heappop(open_heap)
        if current == goal:
            return (_reconstruct(came_from, current), dist)
        if current in closed:
            continue
        closed.add(current)

        for neighbour, cost, _key in graph.adjacency.get(current, []):
            tentative = dist + cost
            if tentative < g_score.get(neighbour, float("inf")):
                came_from[neighbour] = current
                g_score[neighbour] = tentative
                counter += 1
                heapq.heappush(open_heap, (tentative, counter, neighbour))

    return None


def _reconstruct(came_from: Dict[str, str], node: str) -> List[str]:
    """Rebuild the node path by following ``came_from`` back to the start."""
    path = [node]
    while node in came_from:
        node = came_from[node]
        path.append(node)
    path.reverse()
    return path
