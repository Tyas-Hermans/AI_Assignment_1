"""Compute the performance statistics written to ``statistics.txt``.

Definitions are written into the file itself so the output is self-documenting
(the assignment asks us to state how each statistic is computed and the units).
"""

from __future__ import annotations

from typing import List

from .planner import PlanResult

# Fixed metric definitions, echoed into statistics.txt.
_DEFINITIONS = (
    "Definitions:\n"
    "  * Times are in seconds.\n"
    "  * Response time of a handled incident = arrival time - incident timestamp.\n"
    "  * Average response time is the mean response time over handled incidents.\n"
    "  * Deadlines met = incidents whose responder arrived <= deadline, as a\n"
    "    percentage of ALL incidents. The planner also dispatches responders\n"
    "    that will arrive late (a missed deadline is not skipped), so late\n"
    "    arrivals count as missed. An incident that stays unhandled (no capable\n"
    "    responder can reach it) also counts as a missed deadline.\n"
)


def build_statistics(result: PlanResult) -> str:
    """Return the full text of ``statistics.txt`` for a plan result."""
    assignments = result.assignments
    incidents = result.scenario.incidents

    handled = len(assignments)
    responders_used = len({a.responder_id for a in assignments})

    response_times = [a.arrival_time - _timestamp(result, a.incident_id) for a in assignments]
    avg_response = sum(response_times) / handled if handled else 0.0

    met = sum(1 for a in assignments if a.meets_deadline)
    total = len(incidents)
    pct_met = (100.0 * met / total) if total else 0.0

    unhandled = result.unhandled_incident_ids

    lines: List[str] = []
    lines.append("System statistics:")
    lines.append("")
    lines.append(f" * {handled} incidents handled")
    lines.append(f" * {responders_used} responders used")
    lines.append(f" * Average response time: {avg_response:.1f} s")
    lines.append(f" * Deadlines met: {pct_met:.1f}% ({met} of {total} incidents)")
    lines.append(f" * Incidents unhandled: {len(unhandled)} of {len(incidents)}")
    if unhandled:
        lines.append(f"   ({', '.join(unhandled)})")
    lines.append("")
    lines.append(_DEFINITIONS)
    return "\n".join(lines)


def _timestamp(result: PlanResult, incident_id: str) -> float:
    for inc in result.scenario.incidents:
        if inc.id == incident_id:
            return inc.timestamp
    raise KeyError(incident_id)
