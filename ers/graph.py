"""Combine the Bayes and HMM beliefs into a weighted planning graph.

Course concept: this is the bridge from *probabilistic belief* to a concrete
**cost model** the search can optimise.

Each road's expected travel time is

    expected_time = nominal_time · status_multiplier · traffic_multiplier

where ``nominal_time = distance / speed`` (free-flow).

- **status_multiplier** comes from the Bayes belief, conditioned on the road not
  being blocked: Open contributes ×1.0, Congested ×1.5 (fixed by the
  assignment). We condition on "not blocked" because blocked roads are removed
  entirely rather than given a finite cost.
- **traffic_multiplier** comes from the HMM belief, as the expected value of a
  per-level multiplier (Low 1.0, Medium 1.3, High 1.6 — proposed values; traffic
  that is historically heavier should slow a road even when it is Open).
- A road is **removed** from the graph when ``P(Blocked) > blocked_threshold``:
  we refuse to route over roads we believe are impassable.

All multipliers are ≥ 1, which keeps the A* heuristic admissible (see
``search.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from .data_loader import (
    STATUSES,
    TRAFFIC_LEVELS,
    CityGraph,
    Node,
    RoadKey,
    Scenario,
)

Edge = Tuple[str, float, RoadKey]  # (to_node, cost_seconds, road_key)


@dataclass(frozen=True)
class PlanningParams:
    """Tunable cost-model parameters (all defensible defaults)."""

    congested_multiplier: float = 1.5
    blocked_threshold: float = 0.5
    traffic_multiplier: Dict[str, float] = field(
        default_factory=lambda: {"Low": 1.0, "Medium": 1.3, "High": 1.6}
    )


@dataclass
class PlanningGraph:
    """A weighted directed graph of expected travel times.

    Roads believed blocked (``P(Blocked) > threshold``) are excluded from
    ``adjacency`` and recorded in ``blocked`` so the validator can reject any
    plan that uses them.
    """

    nodes: Dict[str, Node]
    adjacency: Dict[str, List[Edge]]
    costs: Dict[RoadKey, float]
    blocked: Set[RoadKey]
    max_speed: float

    def euclidean(self, a: str, b: str) -> float:
        na, nb = self.nodes[a], self.nodes[b]
        return ((na.x - nb.x) ** 2 + (na.y - nb.y) ** 2) ** 0.5

    def heuristic(self, node: str, goal: str) -> float:
        """Admissible A* heuristic: straight-line time at the fastest speed.

        ``euclidean(node, goal) / max_speed`` never overestimates the true
        travel time, because every real path is at least as long as the straight
        line (triangle inequality) and every edge cost is ≥ its free-flow time.
        """
        return self.euclidean(node, goal) / self.max_speed


def expected_status_multiplier(
    status_belief: Dict[str, float], params: PlanningParams
) -> float:
    """Expected speed penalty from road status, conditioned on 'not blocked'.

    ``(P(Open)·1.0 + P(Congested)·1.5) / (P(Open) + P(Congested))``.
    """
    p_open = status_belief["Open"]
    p_cong = status_belief["Congested"]
    denom = p_open + p_cong
    if denom <= 0.0:
        # Degenerate: essentially certainly blocked. Caller drops it anyway;
        # return the worst finite multiplier as a safe fallback.
        return params.congested_multiplier
    return (p_open * 1.0 + p_cong * params.congested_multiplier) / denom


def expected_traffic_multiplier(
    traffic_belief: Dict[str, float], params: PlanningParams
) -> float:
    """Expected traffic penalty = Σ_level P(level) · multiplier(level)."""
    return sum(
        traffic_belief[level] * params.traffic_multiplier[level]
        for level in TRAFFIC_LEVELS
    )


def build_planning_graph(
    scenario: Scenario,
    status_beliefs: Dict[RoadKey, Dict[str, float]],
    traffic_beliefs: Dict[RoadKey, Dict[str, float]],
    params: PlanningParams = PlanningParams(),
) -> PlanningGraph:
    """Assemble the weighted graph used by the route search."""
    city: CityGraph = scenario.graph
    adjacency: Dict[str, List[Edge]] = {node_id: [] for node_id in city.nodes}
    costs: Dict[RoadKey, float] = {}
    blocked: Set[RoadKey] = set()

    for key, road in city.roads.items():
        status = status_beliefs[key]
        if status["Blocked"] > params.blocked_threshold:
            blocked.add(key)
            continue

        status_mult = expected_status_multiplier(status, params)
        traffic_mult = expected_traffic_multiplier(traffic_beliefs[key], params)
        cost = road.nominal_time * status_mult * traffic_mult

        costs[key] = cost
        adjacency[road.frm].append((road.to, cost, key))

    return PlanningGraph(
        nodes=city.nodes,
        adjacency=adjacency,
        costs=costs,
        blocked=blocked,
        max_speed=city.max_speed,
    )
