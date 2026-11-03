"""Unit tests for the Bayesian road-status model."""

import pytest

from ers.bayes import (
    compute_road_status_beliefs,
    marginal_status_prior,
    posterior_with_reports,
)
from ers.data_loader import STATUSES


def _is_distribution(belief):
    return abs(sum(belief.values()) - 1.0) < 1e-9 and all(
        v >= -1e-12 for v in belief.values()
    )


def test_marginal_prior_is_distribution(scenario1):
    prior = marginal_status_prior(scenario1.road_probabilities)
    assert set(prior) == set(STATUSES)
    assert _is_distribution(prior)


def test_marginal_prior_hand_value(scenario1):
    # P(Open) = sum_w P(w) P(Open|w)
    #         = 0.5*0.95 + 0.3*0.60 + 0.2*0.15
    #         = 0.475 + 0.18 + 0.03 = 0.685
    prior = marginal_status_prior(scenario1.road_probabilities)
    assert prior["Open"] == pytest.approx(0.685, abs=1e-9)


def test_posterior_n1_n2_hand_computed(scenario1):
    """Full Bayes posterior for road (N1, N2), verifiable by hand.

    Reports: Citizen=Blocked, Police=Blocked, Sensor=Congested; weather=Rain.
    Prior P(S|Rain): Open=0.6, Congested=0.3, Blocked=0.1.

    Unnormalised:
      Open      = 0.6 * 0.1  * 0.02 * 0.1  = 0.00012
      Congested = 0.3 * 0.2  * 0.08 * 0.8  = 0.00384
      Blocked   = 0.1 * 0.7  * 0.9  * 0.07 = 0.00441
      sum = 0.00837
    """
    reports = scenario1.reports_by_road()[("N1", "N2")]
    post = posterior_with_reports(reports, scenario1.road_probabilities)

    assert _is_distribution(post)
    assert post["Open"] == pytest.approx(0.00012 / 0.00837, abs=1e-6)
    assert post["Congested"] == pytest.approx(0.00384 / 0.00837, abs=1e-6)
    assert post["Blocked"] == pytest.approx(0.00441 / 0.00837, abs=1e-6)
    # Reports strongly suggest the road is blocked.
    assert post["Blocked"] > 0.5


def test_every_road_has_valid_belief(any_scenario):
    beliefs = compute_road_status_beliefs(any_scenario)
    assert set(beliefs) == set(any_scenario.graph.roads)
    for belief in beliefs.values():
        assert _is_distribution(belief)


def test_unreported_road_uses_marginal_prior(scenario1):
    beliefs = compute_road_status_beliefs(scenario1)
    prior = marginal_status_prior(scenario1.road_probabilities)
    reported = set(scenario1.reports_by_road())
    for key, belief in beliefs.items():
        if key not in reported:
            assert belief == pytest.approx(prior)
