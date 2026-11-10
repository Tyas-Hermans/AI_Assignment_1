# Emergency Response System — AI Practicum Assignment 1

**Author:** TODO: full name, student number
**Course:** Artificial Intelligence Practicum — University of Antwerp
**Date:** 12 November 2026
**GitHub repository:** <https://github.com/Tyas-Hermans/AI_Assignment_1>

<div style="page-break-after: always;"></div>

## 1. Approach and implementation

The system reads one scenario folder and runs a fixed pipeline
(`python run.py <scenario> [out]`, all times in **seconds**, distances in metres):

`data_loader` → **Bayes** (road status) + **HMM** (traffic) → `graph` (expected travel
time per road) → **constraint-based allocation** (event-driven simulation, calls **A\***)
→ `plan.json` + `statistics.txt` → `validator`.

**Bayesian reasoning (`ers/bayes.py`).** Model: Weather → Status → {Citizen, Police, Sensor
report}. Given the true status, the reports are treated as conditionally independent, so
P(S | w, reports) ∝ P(S | w) · Π P(report | S). The weather comes from the reports on that
road, because the scenario README says weather is local to the road. A road with no reports
uses P(S) = Σ_w P(w)·P(S | w). Example, worked by hand and checked in a unit test: road
N1→N2 with Rain and reports Blocked/Blocked/Congested gives
P(Open, Congested, Blocked) = (0.014, 0.459, 0.527).

**Hidden Markov Model (`ers/hmm.py`).** For each directed road, the hidden state is the
traffic level (Low/Medium/High). The forward algorithm, renormalised at every step
(filtering), turns the historical observation sequence into P(level_T | obs_1..T). A road
with no observations uses the `initial` distribution. No traffic multipliers are given, so
I chose Low ×1.0, Medium ×1.3, High ×1.6. They are kept ≥ 1 so the A\* heuristic stays
admissible.

**Cost model (`ers/graph.py`).** A road is removed when P(Blocked) > 0.5. Otherwise its
expected time is (distance/speed) · [(P_O·1 + P_C·1.5)/(P_O+P_C)] · Σ P(level)·mult(level).
The status factor is conditioned on "not blocked".

**Search (`ers/search.py`).** A\* with h(n) = euclidean(n, goal) / max_speed. Every edge
costs at least its free-flow time, and on the data every road length is ≥ the straight-line
distance (smallest ratio 1.017). So h is admissible and consistent, and A\* returns optimal
routes. Dijkstra is kept only as a test oracle.

**Constraint-based allocation (`ers/allocation.py`).** An event-driven simulation: decision
moments are incident timestamps and the times responders become free. There is no
look-ahead. At each moment a small CSP is solved:
*variables* = open incidents.
*domain* = free responders that have the required capability and a route, plus
"unassigned".
*constraint* = a responder handles at most one incident.
Backtracking uses **MRV**, value ordering (on time first, then shortest route) and
**forward checking**, which removes a used responder from the other domains. Complete
assignments are compared lexicographically on (priority served, priority served on time,
−total lateness, −total travel), with weights Critical 1000, High 100, Medium 10, Low 1.
The deadline is a **soft** constraint, with a wait rule:
- If no free responder can arrive in time, but a responder that is *busy now* could still
  make the deadline after it becomes free, the incident waits for it. At this moment it may
  only take an on-time responder.
- If nobody can make the deadline any more, the best free responder is sent immediately and
  arrives late.

This uses only information known at that moment, so there is still no look-ahead.

**Assumptions.**
(1) Beliefs are computed once at t = 0 and stay fixed.
(2) A responder is busy from departure until arrival + `time_required`, and then stays at
the incident.
(3) `time` in `plan.json` is the departure time.
(4) A Busy responder leaves from its start location after `time_until_available`.
(5) Deadline feasibility uses *expected* travel times.
(6) Status and traffic factors are independent and multiplied together.

## 2. LLM collaboration

I used **Claude through Claude Code** to design and implement the system. My first
substantive prompt is in the appendix. In it I asked for a design overview first and then a
step-by-step implementation with explanations, so that I could understand and defend the
code.

I tested the scenarios and checked whether I liked the outcome and whether it was correct.
The first time, I did not like it: the system skipped incidents whose deadline could no
longer be met. That is why I sent this follow-up prompt:

```text
- in the statistics van de scenarios staat dat er 100% van de deadlines let zijn maar ook unhandled incidents is meer dan 0                                       
```

As a result, the policy changed from **skip** to **serve late**, and "deadlines met" is now
calculated over *all* incidents.

Later I asked to make deadlines met more important than serving incidents late. The LLM
first tried ranking on-time arrivals higher in the CSP score; this changed nothing. It then
proposed waiting until the deadline had passed before serving late. That made Scenario 2
worse and delayed hopeless incidents for nothing, so I chose its refined version, the wait
rule described in section 1.

## 3. Testing and results

**Unit tests (`tests/`, pytest).**
- Bayes: a hand-calculated posterior, and every belief sums to 1.
- HMM: a hand-calculated one-step belief, the empty sequence falls back to `initial`, and
  repeated High observations drive P(High) up.
- Search: the heuristic is ≤ the true cost for every node pair in all scenarios, A\* cost
  equals Dijkstra cost, paths use only existing non-blocked roads, and a stricter threshold
  removes at least as many roads.
- Allocation: a synthetic CSP where greedy assignment strands a Critical incident but
  backtracking serves both; capability mismatch; late serving; an on-time responder is
  preferred over a late one; an incident waits when a busy responder can still be on time,
  and is served late immediately when nobody can.

TODO: run `pip install -r requirements.txt && pytest -q` and fill in the real count
(expected: 47 passed).

**Plan validation (`ers/validator.py`).** The validator replays `plan.json` in order of
departure time. It checks that:
- the path starts at the responder's current location and ends at the incident;
- every step is an existing, non-blocked road;
- the responder has the required capability;
- start ≥ incident timestamp and start ≥ the time the responder becomes free;
- a responder never has overlapping jobs;
- each incident is assigned at most once.

It also reports whether each deadline is met. The tests go both ways: generated plans pass,
and deliberately corrupted plans fail (wrong capability, non-existent road, blocked road,
duplicate incident, dispatch before the timestamp). The validator also found a real bug:
departure times rounded to 3 decimals made a responder seem to leave about 0.0004 s before
it was free, so Scenario 3 was marked INVALID. Departure times are no longer rounded.

**Scenario results (all plans VALID).**

| Scenario | Handled | Responders used | Avg. response (s) | Deadlines met (final) | Skip policy | Plain serve late |
|---|---|---|---|---|---|---|
| 1 | 7/7 | 4 | 109.9 | 57.1 % (4/7) | 4/7 (3 unhandled) | 4/7 |
| 2 | 12/12 | 9 | 41.2 | 33.3 % (4/12) | 3/12 (9 unhandled) | 4/12 |
| 3 | 20/20 | 9 | 1647.5 | 45.0 % (9/20) | 11/20 (9 unhandled) | 8/20 |

Response time = arrival − incident timestamp, averaged over handled incidents. Deadlines met
is a percentage of **all** incidents; late and unhandled incidents both count as missed.

**What I learned.** Scenario 2 has very tight deadlines. Incident I1 can be reached in
56.2 s against a 55 s deadline, and responder R9 at N1 is cut off by the blocked road N1↔N2.
In Scenario 3, plain serve-late costs on-time arrivals (8 vs 11). Responders sent to
incidents that are already late are still busy when later incidents appear that they could
have reached in time. The wait rule recovers one of these (9/20) while still serving all
20 incidents.

A trade-off I tested: always waiting until the deadline before serving late reaches 11/20
in Scenario 3. However, it lowers Scenario 2 to 3/12 and delays hopeless incidents for
nothing (Scenario 2, I1 arrives at 111 s instead of 56 s).

## 4. Critical reflection

**Limitations and their consequences.**
- *Myopic, irrevocable dispatch.* An incident only waits for responders that are busy
  *now*. If none of them can be on time, it is assigned straight away, even when the
  arrival is far too late (Scenario 3, I6: arrival 5423 s, deadline 2500 s). That responder
  is then unavailable for incidents that have not appeared yet, and an assignment is never
  revised.
- *Blocked risk below the threshold is ignored.* A road with P(Blocked) = 0.49 costs the
  same as a safe road. Deadline checks use expected times, so a plan that is feasible on
  paper can fail in practice.
- *The validator is not independent of the cost model.* It reuses the same Bayes, HMM and
  graph code, so it would not catch a mistake in those modules.
- *The CSP is really exhaustive optimisation.* "Unassigned" is always allowed, so forward
  checking never empties a domain. The run time grows exponentially with the number of open
  incidents, which is fine for these scenario sizes. The weights only guarantee strict
  priority while fewer than 10 incidents of one level are open at the same time.
- *Modelling choices.*
  - Weather is read per road from the reports. The assignment speaks of one known current
    weather.
  - Roads without reports average over the weather prior, giving P(Blocked) = 0.135.
  - The HMM belief is not moved forward in time.
  - The traffic multipliers are my own choice, not derived from data.

**LLM.** The LLM was helpful for structure, implementation and tests. However, its first
policy (skip) produced misleading statistics: 100 % of deadlines met while incidents
remained unhandled. I noticed this only by inspecting the results myself.

**Next improvements, by priority.**
1. Make costs risk-aware, using a P(Blocked) penalty or a chance constraint on arrival.
   *Evaluation:* sample road states from the posteriors (Monte Carlo) and compare the
   share of deadlines actually met.
2. Allow re-planning: cancel a late assignment when a closer responder becomes free, and
   possibly keep a reserve unit per capability for future incidents. *Evaluation:*
   deadlines met and average response time on S1–S3 against the current policy.
3. Give the validator its own independent cost check, and simulate sampled road outcomes.

<div style="page-break-after: always;"></div>

## Appendix — LLM prompts

**First substantive prompt (exact text):**

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

**Follow-up prompt (serve late instead of skip):**

```text
- in the statistics van de scenarios staat dat er 100% van de deadlines let zijn maar ook unhandled incidents is meer dan 0                                       
```
