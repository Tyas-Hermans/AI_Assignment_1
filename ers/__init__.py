"""Emergency Response System (ERS).

A decision-support system that assigns emergency responders to incidents and
plans routes through a city graph under uncertain road conditions.

The package is split into small, single-responsibility modules, each tied to one
AI concept from the course:

- ``data_loader``  : read a scenario folder into typed dataclasses.
- ``bayes``        : Bayesian reasoning over road status (Open/Congested/Blocked).
- ``hmm``          : Hidden Markov Model forward algorithm over traffic levels.
- ``graph``        : combine both beliefs into expected travel times per road.
- ``search``       : A* (and Dijkstra as a test oracle) over the weighted graph.
- ``allocation``   : event-driven simulation + backtracking CSP for assignment.
- ``planner``      : orchestrates the pipeline and produces the plan.
- ``validator``    : independent correctness checks on a produced plan.
- ``stats``        : performance statistics for ``statistics.txt``.
"""
