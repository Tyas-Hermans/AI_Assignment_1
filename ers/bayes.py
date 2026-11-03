"""Bayesian reasoning about road status (Open / Congested / Blocked).

Course concept: **Bayes' rule with conditionally independent observations.**

For each directed road we want the posterior belief over its true status given
the local weather and the noisy reports from citizens, police and sensors.

The generative model behind the data is:

    Weather  ->  Status  ->  { Citizen report, Police report, Sensor report }

so, given the true Status, the three reports are conditionally independent.
By Bayes' rule, for a road whose weather ``w`` is known (reported roads):

    P(S | w, reports) ∝ P(S | w) · Π_r P(report_r | S)

where ``P(report_r | S)`` is read from that source's CPT. For a road with *no*
reports we have no weather and no likelihood, so we marginalise the weather out
of the prior:

    P(S) = Σ_w P(w) · P(S | w)
"""

from __future__ import annotations

from typing import Dict, List

from .data_loader import (
    STATUSES,
    CityGraph,
    RoadKey,
    RoadProbabilities,
    RoadReport,
    Scenario,
    SOURCE_TO_CPT,
)

# A probability distribution over the three statuses.
StatusBelief = Dict[str, float]


def _normalise(weights: Dict[str, float]) -> Dict[str, float]:
    """Scale a dict of non-negative weights to sum to 1.

    If every weight is 0 (which should not happen with valid CPTs) we fall back
    to a uniform distribution so downstream code never divides by zero.
    """
    total = sum(weights.values())
    if total <= 0.0:
        n = len(weights)
        return {k: 1.0 / n for k in weights}
    return {k: v / total for k, v in weights.items()}


def marginal_status_prior(road_probs: RoadProbabilities) -> StatusBelief:
    """P(Status) with the weather marginalised out.

    Used for roads that have no reports: we know nothing about their local
    weather, so we average P(Status | Weather) over the weather prior.
    """
    belief = {s: 0.0 for s in STATUSES}
    for weather, p_w in road_probs.weather_prior.items():
        cond = road_probs.status_given_weather[weather]
        for s in STATUSES:
            belief[s] += p_w * cond[s]
    # Already normalised in theory (priors sum to 1); normalise for safety.
    return _normalise(belief)


def posterior_with_reports(
    reports: List[RoadReport], road_probs: RoadProbabilities
) -> StatusBelief:
    """Posterior P(Status | weather, reports) for a road that has reports.

    The weather is taken from the reports themselves (all reports for a road
    share the same local weather, per the scenario README). The prior is
    P(Status | weather); each report contributes a likelihood factor
    P(reported_status | Status) from its source's CPT.
    """
    weather = reports[0].weather
    prior = road_probs.status_given_weather[weather]

    unnormalised = {s: prior[s] for s in STATUSES}
    for report in reports:
        cpt_name = SOURCE_TO_CPT[report.source]
        cpt = road_probs.report_cpts[cpt_name]  # cpt[true_status][reported]
        for s in STATUSES:
            unnormalised[s] *= cpt[s][report.reported_status]

    return _normalise(unnormalised)


def compute_road_status_beliefs(scenario: Scenario) -> Dict[RoadKey, StatusBelief]:
    """Posterior status belief for every directed road in the graph.

    Roads with reports use :func:`posterior_with_reports`; roads without use the
    weather-marginalised prior :func:`marginal_status_prior`.
    """
    road_probs = scenario.road_probabilities
    reports = scenario.reports_by_road()
    prior = marginal_status_prior(road_probs)  # computed once, reused

    beliefs: Dict[RoadKey, StatusBelief] = {}
    for key in scenario.graph.roads:
        if key in reports:
            beliefs[key] = posterior_with_reports(reports[key], road_probs)
        else:
            beliefs[key] = dict(prior)
    return beliefs
