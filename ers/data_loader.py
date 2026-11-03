"""Load a scenario folder into typed, immutable dataclasses.

This module does *no* reasoning. It only parses the raw CSV/JSON files into
Python objects so the rest of the system can stay clean and testable. Keeping
parsing in one place also means the file formats are documented in exactly one
spot (here).

All units follow the scenario README: distances in metres, speeds in m/s, all
times/durations in seconds.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# A directed road is identified by the (from, to) pair of node ids.
RoadKey = Tuple[str, str]

# The three road statuses and three traffic levels, in a fixed canonical order.
# Fixing the order once here keeps every probability vector consistent across
# the Bayes and HMM modules.
STATUSES: Tuple[str, ...] = ("Open", "Congested", "Blocked")
TRAFFIC_LEVELS: Tuple[str, ...] = ("Low", "Medium", "High")

# Priority ranking used later when the CSP prefers more urgent incidents first.
# Lower number = more urgent.
PRIORITY_RANK: Dict[str, int] = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}

# Maps a report's ``source`` column value to the matching CPT key in
# road_probabilities.json.
SOURCE_TO_CPT: Dict[str, str] = {
    "Citizen": "citizen_report",
    "Police": "police_report",
    "Sensor": "sensor_report",
}


@dataclass(frozen=True)
class Node:
    """A location in the city graph."""

    id: str
    name: str
    x: float
    y: float
    type: str  # e.g. "normal", "hospital"


@dataclass(frozen=True)
class Road:
    """A directed road (edge) with its nominal travel properties.

    ``nominal_time`` is the free-flow travel time (distance / speed) in seconds,
    i.e. the time when the road is Open and traffic is negligible. The graph
    module scales this by the expected multipliers from Bayes and the HMM.
    """

    frm: str
    to: str
    distance: float
    road_type: str
    speed: float

    @property
    def key(self) -> RoadKey:
        return (self.frm, self.to)

    @property
    def nominal_time(self) -> float:
        return self.distance / self.speed


@dataclass(frozen=True)
class Responder:
    """An emergency responder.

    ``free_at`` is the earliest simulation time the responder can depart: 0 if
    Available, otherwise ``time_until_available``. Capabilities are stored as a
    set for fast capability-match checks in the CSP.
    """

    id: str
    type: str
    start_location: str
    capabilities: frozenset
    availability: str  # "Available" or "Busy"
    free_at: float


@dataclass(frozen=True)
class Incident:
    """An incident that becomes known only at ``timestamp``.

    ``deadline`` is an absolute simulation time (the latest arrival time), not a
    duration. ``time_required`` is how long the responder is occupied at the
    scene after arriving.
    """

    id: str
    location: str
    priority: str
    deadline: float
    required_capability: str
    timestamp: float
    time_required: float

    @property
    def priority_rank(self) -> int:
        return PRIORITY_RANK[self.priority]


@dataclass(frozen=True)
class RoadReport:
    """One noisy report about a road's status, from one source, at time 0.

    The ``weather`` is local to the reported road (per the scenario README), so
    it is stored per report rather than globally.
    """

    frm: str
    to: str
    weather: str
    source: str
    reported_status: str


@dataclass(frozen=True)
class TrafficObservation:
    """One historical traffic observation for a directed road.

    ``observation_nr`` orders the sequence for that road; it is NOT a simulation
    timestamp (per the scenario README).
    """

    observation_nr: int
    frm: str
    to: str
    observation: str


@dataclass(frozen=True)
class RoadProbabilities:
    """Conditional probability tables for the Bayesian road-status model."""

    weather_prior: Dict[str, float]                      # P(Weather)
    status_given_weather: Dict[str, Dict[str, float]]    # P(Status | Weather)
    # P(Reported | Status) for each source, keyed by CPT name
    # (citizen_report / police_report / sensor_report).
    report_cpts: Dict[str, Dict[str, Dict[str, float]]]


@dataclass(frozen=True)
class TrafficProbabilities:
    """Parameters of the traffic HMM (hidden state = true traffic level)."""

    initial: Dict[str, float]                       # P(state at step 1)
    transition: Dict[str, Dict[str, float]]         # P(next | current)
    emission: Dict[str, Dict[str, float]]           # P(observation | state)


@dataclass
class CityGraph:
    """The city as a directed graph, with convenience lookups."""

    nodes: Dict[str, Node]
    roads: Dict[RoadKey, Road]
    # Adjacency: node id -> list of outgoing road keys.
    adjacency: Dict[str, List[RoadKey]] = field(default_factory=dict)

    def neighbours(self, node_id: str) -> List[RoadKey]:
        return self.adjacency.get(node_id, [])

    def euclidean(self, a: str, b: str) -> float:
        """Straight-line distance in metres between two nodes.

        Used by the A* heuristic. It is a lower bound on any real road distance
        between the nodes (triangle inequality), which is what makes the
        heuristic admissible.
        """
        na, nb = self.nodes[a], self.nodes[b]
        return math.hypot(na.x - nb.x, na.y - nb.y)

    @property
    def max_speed(self) -> float:
        """Fastest speed in the network (for the A* heuristic denominator)."""
        return max((r.speed for r in self.roads.values()), default=1.0)


@dataclass
class Scenario:
    """Everything loaded from one scenario folder."""

    name: str
    graph: CityGraph
    responders: List[Responder]
    incidents: List[Incident]
    road_reports: List[RoadReport]
    traffic_observations: List[TrafficObservation]
    road_probabilities: RoadProbabilities
    traffic_probabilities: TrafficProbabilities

    def reports_by_road(self) -> Dict[RoadKey, List[RoadReport]]:
        """Group reports by the directed road they describe."""
        grouped: Dict[RoadKey, List[RoadReport]] = {}
        for r in self.road_reports:
            grouped.setdefault((r.frm, r.to), []).append(r)
        return grouped

    def observations_by_road(self) -> Dict[RoadKey, List[TrafficObservation]]:
        """Group observations by road, sorted by ``observation_nr``."""
        grouped: Dict[RoadKey, List[TrafficObservation]] = {}
        for o in self.traffic_observations:
            grouped.setdefault((o.frm, o.to), []).append(o)
        for key in grouped:
            grouped[key].sort(key=lambda o: o.observation_nr)
        return grouped


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #

def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_graph(path: str) -> CityGraph:
    raw = _load_json(path)
    nodes = {
        n["id"]: Node(
            id=n["id"],
            name=n.get("name", n["id"]),
            x=float(n["x"]),
            y=float(n["y"]),
            type=n.get("type", "normal"),
        )
        for n in raw["nodes"]
    }
    roads: Dict[RoadKey, Road] = {}
    adjacency: Dict[str, List[RoadKey]] = {node_id: [] for node_id in nodes}
    for r in raw["roads"]:
        road = Road(
            frm=r["from"],
            to=r["to"],
            distance=float(r["distance"]),
            road_type=r.get("road_type", "secondary"),
            speed=float(r["speed"]),
        )
        roads[road.key] = road
        adjacency.setdefault(road.frm, []).append(road.key)
    return CityGraph(nodes=nodes, roads=roads, adjacency=adjacency)


def _load_responders(path: str) -> List[Responder]:
    responders: List[Responder] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            availability = row["availability"].strip()
            time_until = float(row["time_until_available"])
            # Available responders can leave at t=0; busy ones at time_until.
            free_at = time_until if availability == "Busy" else 0.0
            caps = frozenset(
                c.strip() for c in row["capabilities"].split(";") if c.strip()
            )
            responders.append(
                Responder(
                    id=row["responder_id"].strip(),
                    type=row["type"].strip(),
                    start_location=row["start_location"].strip(),
                    capabilities=caps,
                    availability=availability,
                    free_at=free_at,
                )
            )
    return responders


def _load_incidents(path: str) -> List[Incident]:
    incidents: List[Incident] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            incidents.append(
                Incident(
                    id=row["incident_id"].strip(),
                    location=row["location"].strip(),
                    priority=row["priority"].strip(),
                    deadline=float(row["deadline"]),
                    required_capability=row["required_capability"].strip(),
                    timestamp=float(row["timestamp"]),
                    time_required=float(row["time_required"]),
                )
            )
    return incidents


def _load_road_reports(path: str) -> List[RoadReport]:
    reports: List[RoadReport] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            reports.append(
                RoadReport(
                    frm=row["from"].strip(),
                    to=row["to"].strip(),
                    weather=row["weather"].strip(),
                    source=row["source"].strip(),
                    reported_status=row["reported_status"].strip(),
                )
            )
    return reports


def _load_traffic_observations(path: str) -> List[TrafficObservation]:
    observations: List[TrafficObservation] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            observations.append(
                TrafficObservation(
                    observation_nr=int(row["observation_nr"]),
                    frm=row["from"].strip(),
                    to=row["to"].strip(),
                    observation=row["traffic_observation"].strip(),
                )
            )
    return observations


def _load_road_probabilities(path: str) -> RoadProbabilities:
    raw = _load_json(path)
    return RoadProbabilities(
        weather_prior=raw["weather_prior"],
        status_given_weather=raw["road_status_given_weather"],
        report_cpts={
            "citizen_report": raw["citizen_report"],
            "police_report": raw["police_report"],
            "sensor_report": raw["sensor_report"],
        },
    )


def _load_traffic_probabilities(path: str) -> TrafficProbabilities:
    raw = _load_json(path)
    return TrafficProbabilities(
        initial=raw["initial"],
        transition=raw["transition"],
        emission=raw["emission"],
    )


def load_scenario(folder: str) -> Scenario:
    """Load every file in a scenario folder into a :class:`Scenario`.

    Parameters
    ----------
    folder:
        Path to a scenario directory containing ``city_graph.json``,
        ``responders.csv``, ``incidents.csv``, ``road_reports.csv``,
        ``observed_traffic.csv``, ``road_probabilities.json`` and
        ``traffic_probabilities.json``.
    """
    folder = os.path.abspath(folder)
    name = os.path.basename(folder.rstrip(os.sep))

    def p(filename: str) -> str:
        return os.path.join(folder, filename)

    return Scenario(
        name=name,
        graph=_load_graph(p("city_graph.json")),
        responders=_load_responders(p("responders.csv")),
        incidents=_load_incidents(p("incidents.csv")),
        road_reports=_load_road_reports(p("road_reports.csv")),
        traffic_observations=_load_traffic_observations(p("observed_traffic.csv")),
        road_probabilities=_load_road_probabilities(p("road_probabilities.json")),
        traffic_probabilities=_load_traffic_probabilities(p("traffic_probabilities.json")),
    )
