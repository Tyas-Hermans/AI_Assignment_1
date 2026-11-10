"""Assign responders to incidents over time.

Course concept: **constraint satisfaction (backtracking + MRV + forward
checking)** driven by an **event-based simulation** that respects the
no-look-ahead rule.

Two layers:

1. *Event-driven simulation* (`run_simulation`). Incidents become known only at
   their timestamp and responders free up only at ``arrival + time_required``.
   We process decision moments in increasing time; at each one we know only what
   has happened so far — never the future.

2. *CSP at each decision moment* (`solve_event_csp`). Variables are the
   currently-open incidents; each variable's domain is the set of free responders
   that have the required capability and a route to the incident (plus the
   option of leaving it unassigned). The deadline is a *soft* constraint: a
   responder that would arrive late is still a valid value, it is just ranked
   below on-time responders. The only hard constraint between variables is that
   a responder serves at most one incident. We solve with backtracking, using
   MRV for variable ordering, value ordering (on-time first, then shortest
   route), and forward checking to prune a responder from other domains once it
   is used. Among complete assignments we keep the best by, in order:
   served priority, priority served *on time*, total lateness (lower is
   better), total response time (lower is better).

Policy for incidents no free, capable responder can reach in time: **wait if
on time is still possible, otherwise serve late.**

- *Wait:* if a responder that is busy right now could still arrive on time
  after it becomes free (its free time + route time ≤ deadline), the incident
  only accepts on-time responders at this moment. It stays open and is
  reconsidered at the next decision moment (e.g. when that responder frees).
  This keeps free responders available for incidents they can reach on time,
  instead of tying them up on a late job.
- *Serve late:* if no responder — free or busy — can still make the deadline,
  the best free one is dispatched immediately; waiting would only make the
  arrival later.

This uses only what is known at time ``now`` (current busy times and
locations), so the no-look-ahead rule still holds. An incident ends unhandled
only if no capable responder can ever reach it (no matching capability, or
every route is blocked).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .data_loader import Incident, Responder, Scenario
from .graph import PlanningGraph
from .search import a_star

# Priority weights used when scoring a set of served incidents. A more urgent
# incident is worth strictly more than any number of less urgent ones at the
# next level down in these scenarios (domains are small), which encodes "prefer
# higher priority".
PRIORITY_WEIGHT: Dict[str, float] = {
    "Critical": 1000.0,
    "High": 100.0,
    "Medium": 10.0,
    "Low": 1.0,
}


@dataclass
class Assignment:
    """A single responder→incident assignment with its route and timing."""

    responder_id: str
    incident_id: str
    path: List[str]
    start_time: float      # departure time = the decision moment ("time" in plan.json)
    travel_time: float     # expected travel time along the route
    arrival_time: float    # start_time + travel_time
    deadline: float        # the incident's absolute deadline (for stats/validation)

    @property
    def meets_deadline(self) -> bool:
        return self.arrival_time <= self.deadline + 1e-9


@dataclass
class AllocationResult:
    assignments: List[Assignment]
    unhandled_incident_ids: List[str]


# A route candidate for one (responder, incident) pair at a decision moment.
@dataclass
class _Candidate:
    responder_id: str
    path: List[str]
    travel_time: float
    arrival_time: float
    lateness: float        # max(0, arrival_time - deadline); 0 means on time

    @property
    def on_time(self) -> bool:
        return self.lateness <= 1e-9


class _RouteCache:
    """Memoises A* routes; the graph is static so (from, to) → route is stable."""

    def __init__(self, graph: PlanningGraph):
        self._graph = graph
        self._cache: Dict[Tuple[str, str], Optional[Tuple[List[str], float]]] = {}

    def route(self, frm: str, to: str) -> Optional[Tuple[List[str], float]]:
        if (frm, to) not in self._cache:
            self._cache[(frm, to)] = a_star(self._graph, frm, to)
        return self._cache[(frm, to)]


def _candidates_for_incident(
    incident: Incident,
    free_responder_ids: List[str],
    locations: Dict[str, str],
    responders_by_id: Dict[str, Responder],
    now: float,
    routes: _RouteCache,
    on_time_only: bool = False,
) -> List[_Candidate]:
    """Feasible responders for one incident at time ``now``.

    A responder is feasible if it has the required capability and a route exists
    to the incident. Arriving after the deadline does *not* exclude it (soft
    constraint); its lateness is recorded instead — unless ``on_time_only`` is
    set, which is used when the incident should wait for a busy responder that
    can still be on time. Ordered on-time first, then by shortest route.
    """
    out: List[_Candidate] = []
    for rid in free_responder_ids:
        responder = responders_by_id[rid]
        if incident.required_capability not in responder.capabilities:
            continue
        route = routes.route(locations[rid], incident.location)
        if route is None:
            continue
        path, travel = route
        arrival = now + travel
        lateness = max(0.0, arrival - incident.deadline)
        if on_time_only and lateness > 1e-9:
            continue  # this incident waits for an on-time responder instead
        out.append(_Candidate(rid, path, travel, arrival, lateness))
    # On-time candidates first, then shortest response.
    out.sort(key=lambda c: (not c.on_time, c.travel_time))
    return out


def solve_event_csp(
    open_incidents: List[Incident],
    free_responder_ids: List[str],
    locations: Dict[str, str],
    responders_by_id: Dict[str, Responder],
    now: float,
    routes: _RouteCache,
    waiting_ids: frozenset = frozenset(),
) -> Dict[str, _Candidate]:
    """Backtracking CSP: choose responders for the open incidents at time ``now``.

    ``waiting_ids`` are incidents that may only take an on-time responder this
    round (a busy responder could still reach them on time later).

    Returns a mapping ``incident_id -> chosen _Candidate`` for the incidents that
    get served this round. Incidents left out are simply not in the mapping.
    """
    # Build each incident's domain (feasible candidates) once; forward checking
    # will prune responders from these as the search assigns them.
    domains: Dict[str, List[_Candidate]] = {
        inc.id: _candidates_for_incident(
            inc, free_responder_ids, locations, responders_by_id, now, routes,
            on_time_only=inc.id in waiting_ids,
        )
        for inc in open_incidents
    }
    incident_by_id = {inc.id: inc for inc in open_incidents}

    best_choice: Dict[str, _Candidate] = {}
    best_score: Tuple[float, float, float, float] = (-1.0, 0.0, 0.0, 0.0)

    def score(choice: Dict[str, _Candidate]) -> Tuple[float, float, float, float]:
        # Compared lexicographically (tuples compare element by element):
        # 1. serve as much (priority-weighted) as possible, late or not;
        # 2. of that, have as much priority as possible arrive on time;
        # 3. then minimise total lateness, 4. then total travel.
        # Quantities to minimise are negated so that "bigger is better".
        served = sum(PRIORITY_WEIGHT[incident_by_id[iid].priority] for iid in choice)
        on_time = sum(
            PRIORITY_WEIGHT[incident_by_id[iid].priority]
            for iid, c in choice.items()
            if c.on_time
        )
        total_lateness = sum(c.lateness for c in choice.values())
        total_travel = sum(c.travel_time for c in choice.values())
        return (served, on_time, -total_lateness, -total_travel)

    def backtrack(
        remaining: List[str], used: set, choice: Dict[str, _Candidate]
    ) -> None:
        nonlocal best_choice, best_score
        if not remaining:
            s = score(choice)
            if s > best_score:
                best_score = s
                best_choice = dict(choice)
            return

        # MRV: branch on the incident with the fewest still-available candidates.
        def available(iid: str) -> List[_Candidate]:
            return [c for c in domains[iid] if c.responder_id not in used]

        iid = min(remaining, key=lambda i: len(available(i)))
        rest = [i for i in remaining if i != iid]

        # Value ordering: try assigning feasible responders (on time, then shortest)
        # via forward checking, then the option of leaving this incident unassigned.
        for cand in available(iid):
            choice[iid] = cand
            used.add(cand.responder_id)
            backtrack(rest, used, choice)
            used.discard(cand.responder_id)
            del choice[iid]

        # Leave this incident unassigned this round.
        backtrack(rest, used, choice)

    backtrack(list(domains.keys()), set(), {})
    return best_choice


def busy_responder_can_be_on_time(
    incident: Incident,
    free_at: Dict[str, float],
    locations: Dict[str, str],
    responders_by_id: Dict[str, Responder],
    now: float,
    routes: _RouteCache,
) -> bool:
    """True if a currently busy, capable responder could still arrive on time.

    For each responder that is busy at ``now`` we know when it becomes free and
    where it will be (the location of its current job). If leaving from there at
    that time still reaches the incident by its deadline, the incident is worth
    waiting for. Uses only information known at ``now`` (no look-ahead).
    """
    for rid, t_free in free_at.items():
        if t_free <= now:
            continue  # free responders are handled by the CSP itself
        if incident.required_capability not in responders_by_id[rid].capabilities:
            continue
        route = routes.route(locations[rid], incident.location)
        if route is not None and t_free + route[1] <= incident.deadline + 1e-9:
            return True
    return False


def run_simulation(scenario: Scenario, graph: PlanningGraph) -> AllocationResult:
    """Event-driven allocation over the whole incident stream."""
    responders_by_id = {r.id: r for r in scenario.responders}
    incidents = scenario.incidents
    routes = _RouteCache(graph)

    # Mutable responder state: when each is next free, and where it currently is.
    free_at: Dict[str, float] = {r.id: r.free_at for r in scenario.responders}
    locations: Dict[str, str] = {r.id: r.start_location for r in scenario.responders}

    assigned_incident_ids: set = set()
    assignments: List[Assignment] = []

    # Decision moments: incident arrivals and responder free-up times. New
    # free-up times are pushed as assignments are made.
    event_heap: List[float] = []
    seen_times: set = set()

    def push_time(t: float) -> None:
        if t not in seen_times:
            seen_times.add(t)
            heapq.heappush(event_heap, t)

    for inc in incidents:
        push_time(inc.timestamp)
    for r in scenario.responders:
        push_time(r.free_at)

    while event_heap:
        now = heapq.heappop(event_heap)

        open_incidents = [
            inc
            for inc in incidents
            if inc.timestamp <= now and inc.id not in assigned_incident_ids
        ]
        if not open_incidents:
            continue

        free_responder_ids = [rid for rid, t in free_at.items() if t <= now]
        if not free_responder_ids:
            continue

        # Incidents that should wait: a busy responder can still be on time.
        # Waiting is safe because that responder's free-up time is already a
        # decision moment, so the incident will be reconsidered then.
        waiting_ids = frozenset(
            inc.id
            for inc in open_incidents
            if busy_responder_can_be_on_time(
                inc, free_at, locations, responders_by_id, now, routes
            )
        )

        choice = solve_event_csp(
            open_incidents, free_responder_ids, locations, responders_by_id, now,
            routes, waiting_ids,
        )

        for iid, cand in choice.items():
            incident = next(i for i in incidents if i.id == iid)
            assignments.append(
                Assignment(
                    responder_id=cand.responder_id,
                    incident_id=iid,
                    path=cand.path,
                    start_time=now,
                    travel_time=cand.travel_time,
                    arrival_time=cand.arrival_time,
                    deadline=incident.deadline,
                )
            )
            assigned_incident_ids.add(iid)
            # The responder is occupied until it finishes, then stays on scene.
            finish = cand.arrival_time + incident.time_required
            free_at[cand.responder_id] = finish
            locations[cand.responder_id] = incident.location
            push_time(finish)  # it may take another incident after finishing

    unhandled = [i.id for i in incidents if i.id not in assigned_incident_ids]
    return AllocationResult(assignments=assignments, unhandled_incident_ids=unhandled)
