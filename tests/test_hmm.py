"""Unit tests for the traffic HMM (forward algorithm)."""

import pytest

from ers.hmm import compute_traffic_beliefs, forward_belief
from ers.data_loader import TRAFFIC_LEVELS


def _is_distribution(belief):
    return abs(sum(belief.values()) - 1.0) < 1e-9 and all(
        v >= -1e-12 for v in belief.values()
    )


def test_empty_sequence_returns_initial(scenario1):
    belief = forward_belief([], scenario1.traffic_probabilities)
    assert belief == pytest.approx(scenario1.traffic_probabilities.initial)


def test_single_observation_hand_value(scenario1):
    # One "High" observation: belief ∝ initial(s) * emission(s, High).
    #   Low:    0.6 * 0.01 = 0.006
    #   Medium: 0.3 * 0.10 = 0.030
    #   High:   0.1 * 0.80 = 0.080
    #   sum = 0.116
    belief = forward_belief(["High"], scenario1.traffic_probabilities)
    assert _is_distribution(belief)
    assert belief["High"] == pytest.approx(0.080 / 0.116, abs=1e-9)
    assert belief["Medium"] == pytest.approx(0.030 / 0.116, abs=1e-9)
    assert belief["Low"] == pytest.approx(0.006 / 0.116, abs=1e-9)


def test_beliefs_always_sum_to_one(scenario1):
    tp = scenario1.traffic_probabilities
    for seq in ([], ["Low"], ["High", "High", "High"], ["Low", "Medium", "High", "Low"]):
        assert _is_distribution(forward_belief(seq, tp))


def test_repeated_high_pushes_belief_to_high(scenario1):
    tp = scenario1.traffic_probabilities
    belief = forward_belief(["High"] * 6, tp)
    assert belief["High"] == max(belief.values())
    assert belief["High"] > 0.5


def test_every_road_has_valid_belief(any_scenario):
    beliefs = compute_traffic_beliefs(any_scenario)
    assert set(beliefs) == set(any_scenario.graph.roads)
    for belief in beliefs.values():
        assert set(belief) == set(TRAFFIC_LEVELS)
        assert _is_distribution(belief)
