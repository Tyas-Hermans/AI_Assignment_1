# Project: AI Assignment 1

## LLM collaboration

- **LLM used:** Claude, through Claude Code.
- **First substantive prompt** (sent exactly as shown below):

```text
I am working on a university AI practicum assignment (attached: Assignment1.pdf, plus a
Scenarios folder with Scenario1–3 and a README.md describing units). Please help me
design and implement it in Python. Read the PDF and the scenario README first; they are
the authoritative specification. Below is a summary of what I need and the design
direction I want to take.

GOAL
A decision-support system that, for one scenario folder, assigns emergency responders
to incidents and plans routes through the city graph under uncertain road conditions.
Entry point: `python run.py <scenario_folder>` writes plan.json and statistics.txt
(format exactly as in section 5 of the PDF). All times are in seconds, distances in
metres, speeds in m/s.

REQUIRED AI COMPONENTS (implement them myself, as taught in the course — no solver or
ML libraries; standard library plus numpy at most)
1. Bayesian reasoning (road_reports.csv + road_probabilities.json): for each road,
   compute the posterior P(Status | weather, reports) over Open/Congested/Blocked,
   treating reports from different sources as conditionally independent given the
   true status. Roads without reports have no known weather, so marginalise over the
   weather prior.
2. Hidden Markov Model (observed_traffic.csv + traffic_probabilities.json): per directed
   road, run the forward algorithm over its observation sequence to get a belief over
   the current traffic level (Low/Medium/High). Roads without observations get a
   documented default. Traffic multipliers are not given — propose simple values and
   justify them.
3. Search: combine both into an expected travel time per road
   (distance/speed × expected multiplier; Congested = ×1.5) and decide how likely-blocked
   roads are handled (e.g. a threshold on P(Blocked) — make it a parameter). Use A* with
   a straight-line-distance / max-speed heuristic and verify it is admissible.
4. Constraint-based allocation: an event-driven simulation in which incidents only
   become known at their timestamp (no look-ahead). At each decision moment, solve a
   small CSP: variables = open incidents, domains = responders that have the required
   capability and are free (respecting Busy + time_until_available). Constraints: one
   incident per responder at a time, capability match, arrival before the absolute
   deadline. Use backtracking with simple heuristics (e.g. MRV, forward checking) and
   prefer higher priority and shorter response times. Incidents that cannot be served
   in time must be handled explicitly (state the policy: skip, or serve late).

ASSUMPTIONS TO STATE EXPLICITLY (choose reasonable ones; put them in one list)
- Road states are static during the simulation; a responder stays at the incident
  location after finishing; a responder is occupied from departure until
  arrival + time_required; "time" in plan.json is the departure time; how deadline
  feasibility is judged when travel times are only expected values.

VALIDATION AND STATISTICS
- A separate validator that checks plan.json: path starts at the responder's current
  location and ends at the incident location, every step is an existing road, no road
  treated as blocked is used, capability matches, start time ≥ incident timestamp and
  ≥ responder free time, no overlapping assignments per responder, each incident
  assigned at most once, deadline met or not (report which).
- statistics.txt: incidents handled, responders used, average response time
  (define it: arrival time − incident timestamp), % deadlines met, with unhandled
  incidents counted and listed. Write the definitions into the file.

TESTS
Small pytest unit tests per component (a Bayes posterior I can check by hand, HMM
beliefs sum to 1, A* gives the same cost as Dijkstra, CSP respects capabilities and
availability), plus running all three scenarios through the validator.

SCOPE AND STYLE
Keep it simple and readable rather than optimal: a few clearly separated modules,
docstrings that link each part to the course concept, a README.md with install/run
instructions. Do not write the report for me. I need to understand and defend every
line, so:
1. First give a short design overview (modules, data flow, assumptions) and wait for
   my OK.
2. Then implement step by step, explaining the key decisions and where the approach
   is a simplification or might be wrong.
```
- this is a followup prompt i send because i didn't like that it skipped tasks after missing the deadline