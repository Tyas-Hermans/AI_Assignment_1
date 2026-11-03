"""Hidden Markov Model over traffic levels (Low / Medium / High).

Course concept: **HMM filtering with the forward algorithm.**

For each directed road we have a historical sequence of noisy traffic
observations. The true traffic level is hidden and evolves as a Markov chain;
each step emits a noisy observation. We want the belief over the *current*
(most recent) traffic level given the whole observation sequence:

    P(state_T | obs_1..T)

The forward algorithm computes this incrementally. With the forward message
``alpha`` renormalised at every step (filtering form):

    alpha_1(s)  ∝ initial(s) · emission(s, obs_1)
    alpha_t(s)  ∝ emission(s, obs_t) · Σ_s' alpha_{t-1}(s') · transition(s', s)

The belief over the current level is ``alpha_T`` (already normalised).

Roads with no observations get a documented default: the HMM's ``initial``
distribution. (An alternative default is the chain's stationary distribution;
``initial`` is chosen for simplicity and because it is given directly.)
"""

from __future__ import annotations

from typing import Dict, List

from .data_loader import (
    TRAFFIC_LEVELS,
    RoadKey,
    Scenario,
    TrafficObservation,
    TrafficProbabilities,
)

# A probability distribution over the three traffic levels.
TrafficBelief = Dict[str, float]


def _normalise(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values())
    if total <= 0.0:
        n = len(weights)
        return {k: 1.0 / n for k in weights}
    return {k: v / total for k, v in weights.items()}


def forward_belief(
    observations: List[str], traffic_probs: TrafficProbabilities
) -> TrafficBelief:
    """Run the forward algorithm over an observation sequence.

    Parameters
    ----------
    observations:
        The traffic observations (e.g. ``["Low", "Low", "Medium"]``) in
        sequence order. If empty, the ``initial`` distribution is returned.
    """
    initial = traffic_probs.initial
    transition = traffic_probs.transition
    emission = traffic_probs.emission

    if not observations:
        # No evidence for this road: fall back to the prior over the state.
        return _normalise({s: initial[s] for s in TRAFFIC_LEVELS})

    # Step 1: prior times the first emission.
    first = observations[0]
    alpha = {s: initial[s] * emission[s][first] for s in TRAFFIC_LEVELS}
    alpha = _normalise(alpha)

    # Steps 2..T: predict through the transition model, then weight by emission.
    for obs in observations[1:]:
        predicted = {
            s: sum(alpha[s_prev] * transition[s_prev][s] for s_prev in TRAFFIC_LEVELS)
            for s in TRAFFIC_LEVELS
        }
        alpha = {s: predicted[s] * emission[s][obs] for s in TRAFFIC_LEVELS}
        alpha = _normalise(alpha)

    return alpha


def compute_traffic_beliefs(scenario: Scenario) -> Dict[RoadKey, TrafficBelief]:
    """Current-traffic belief for every directed road in the graph.

    Roads with observations use the forward algorithm; roads without use the
    ``initial`` distribution default.
    """
    traffic_probs = scenario.traffic_probabilities
    obs_by_road = scenario.observations_by_road()
    default = _normalise({s: traffic_probs.initial[s] for s in TRAFFIC_LEVELS})

    beliefs: Dict[RoadKey, TrafficBelief] = {}
    for key in scenario.graph.roads:
        sequence = obs_by_road.get(key)
        if sequence:
            observations = [o.observation for o in sequence]
            beliefs[key] = forward_belief(observations, traffic_probs)
        else:
            beliefs[key] = dict(default)
    return beliefs
