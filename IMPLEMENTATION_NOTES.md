# Implementation notes

A running, step-by-step explanation of how the Emergency Response System (ERS)
is built: what each module does, which course concept it implements, and which
design decisions are simplifications that could be wrong. This is reference
material for understanding and defending the code — it is **not** the report.

Units everywhere (from the scenario README): distances in **metres**, speeds in
**m/s**, all times/durations in **seconds**.

## Architecture at a glance

```
run.py              entry point: python run.py <scenario_folder>
ers/data_loader.py  read a scenario folder into typed dataclasses
ers/bayes.py        P(Status | weather, reports)           [Bayesian reasoning]
ers/hmm.py          forward algorithm over traffic levels   [Hidden Markov Model]
ers/graph.py        expected travel time per road; drop likely-blocked roads
ers/search.py       A* planning + Dijkstra oracle           [Search]
ers/allocation.py   event-driven simulation + backtracking  [CSP allocation]
ers/planner.py      orchestrates the whole pipeline
ers/validator.py    independent correctness checks on a plan
ers/stats.py        performance statistics
```

Data flow:

```
scenario folder
   └─ data_loader ─┬─ bayes (road-status belief) ─┐
                   └─ hmm   (traffic belief)       ├─ graph (expected times)
                                                   │        │
                                                   │        ▼
                                                   │   allocation ── calls ──> search (A*)
                                                   │        │
                                                   ▼        ▼
                                              plan.json + statistics.txt
                                                   │
                                              validator (reads plan.json independently)
```

## Agreed design parameters

- **Unhandled incidents:** ~~skip them~~ → changed to **serve late** (see
  "Change: serve late instead of skip" at the end of this file).
- **Traffic multipliers:** Low = 1.0, Medium = 1.3, High = 1.6.
- **Blocked threshold:** a road is removed from the graph when
  P(Blocked) > 0.5.
- **Congested multiplier:** ×1.5 (fixed by the assignment).

All of these are exposed as named constants so they can be tuned and defended.

## Stated assumptions

1. Road-condition beliefs are computed **once at t = 0** and treated as static
   during the simulation (the README says road reports are available at time 0
   and traffic observations are historical).
2. A responder **stays at the incident location** after finishing.
3. A responder is **occupied from departure until arrival + time_required**.
4. `"time"` in `plan.json` is the **departure** time.
5. Deadline feasibility is judged on **expected** travel times, so a plan can be
   feasible in expectation yet miss its deadline under an unlucky road draw.
6. **No look-ahead:** an incident exists for the planner only from its timestamp.

---

## Batch 1 — project skeleton + data loading

**Files:** `ers/__init__.py`, `ers/data_loader.py`, plus `Scenarios/` extracted
to the repo root so the system is self-contained and runs as
`python run.py Scenarios/Scenario1`.

### What `data_loader.py` does

It parses the seven scenario files into immutable dataclasses and does **no
reasoning** — keeping all file-format knowledge in one place so the AI modules
stay clean and unit-testable.

| File | Dataclass(es) | Notes |
|------|---------------|-------|
| `city_graph.json` | `Node`, `Road`, `CityGraph` | roads are **directed**; each direction is a separate edge in the data |
| `responders.csv` | `Responder` | `capabilities` split on `;` into a set; `free_at` = 0 if Available else `time_until_available` |
| `incidents.csv` | `Incident` | `deadline` stored as an **absolute** time; `priority_rank` maps Critical→0 … Low→3 |
| `road_reports.csv` | `RoadReport` | `weather` stored **per report** (weather is local to a road) |
| `observed_traffic.csv` | `TrafficObservation` | kept as a sequence; sorted by `observation_nr` on demand |
| `road_probabilities.json` | `RoadProbabilities` | weather prior, P(Status\|Weather), and one report CPT per source |
| `traffic_probabilities.json` | `TrafficProbabilities` | HMM initial / transition / emission tables |

### Key decisions

- **Canonical orderings** `STATUSES` and `TRAFFIC_LEVELS` are defined once so every
  probability vector in the Bayes and HMM modules lines up consistently.
- **`free_at`** turns the `availability` + `time_until_available` pair into a single
  number the scheduler can compare against, so later modules never re-interpret the
  "Available/Busy" string.
- **`Road.nominal_time` = distance / speed** is the free-flow travel time; the graph
  module multiplies it by the expected Bayes/HMM factors. Storing it here keeps the
  physics in one obvious place.
- **`CityGraph.euclidean`** and **`max_speed`** exist specifically to support the
  admissible A* heuristic in Batch 3.
- **Grouping helpers** (`reports_by_road`, `observations_by_road`) prepare exactly the
  shape the Bayes and HMM modules need (observations arrive pre-sorted by
  `observation_nr`, which the README stresses is a sequence index, not a timestamp).

### Verified

Loads Scenario 1/2/3 without error. Scenario sizes: 5/12/26 nodes, 14/30/60 roads,
4/10/10 responders, 7/12/20 incidents. `max_speed` = 30 m/s in all three (used by
the heuristic). Spot-checked that reports on `(N1,N2)` and the observation sequence
on `(H1,N1)` parse correctly.

---

## Batch 2 — belief components: Bayes + HMM

**Files:** `ers/bayes.py`, `ers/hmm.py`, `tests/test_bayes.py`, `tests/test_hmm.py`,
`tests/conftest.py`.

### `bayes.py` — road status belief *(Bayesian reasoning)*

Generative model: `Weather → Status → {Citizen, Police, Sensor reports}`. Given the
true status the three reports are **conditionally independent**, so Bayes' rule gives

```
reported road:   P(S | w, reports) ∝ P(S | w) · Π_r P(report_r | S)
unreported road: P(S)              = Σ_w P(w) · P(S | w)      (weather marginalised)
```

- `P(S | w)` is `road_status_given_weather[w]`; `P(report_r | S)` is read from the
  source's CPT, indexed `cpt[true_status][reported_status]`.
- `posterior_with_reports` multiplies the weather-conditioned prior by one likelihood
  factor per report, then normalises. The weather is taken from the reports (weather
  is local to a road, and all its reports agree on it).
- `marginal_status_prior` handles roads with no reports — it averages
  `P(S | weather)` over the weather prior. Computed once and reused.
- `compute_road_status_beliefs` returns a belief for **every** road in the graph.

**Hand-check (unit test).** Road `(N1,N2)`, weather Rain, reports
Citizen=Blocked / Police=Blocked / Sensor=Congested:

| S | prior·likelihoods | unnorm | posterior |
|---|---|---|---|
| Open | 0.6·0.1·0.02·0.1 | 0.00012 | 0.0143 |
| Congested | 0.3·0.2·0.08·0.8 | 0.00384 | 0.4588 |
| Blocked | 0.1·0.7·0.9·0.07 | 0.00441 | 0.5269 |

So `P(Blocked) ≈ 0.527 > 0.5` → this road will be **dropped** from the graph in
Batch 3. The test asserts these exact values.

### `hmm.py` — traffic belief *(Hidden Markov Model, forward algorithm)*

Hidden state = true traffic level (Low/Medium/High); each historical observation is a
noisy emission. We want the belief over the **current** level:

```
alpha_1(s) ∝ initial(s) · emission(s, obs_1)
alpha_t(s) ∝ emission(s, obs_t) · Σ_s' alpha_{t-1}(s') · transition(s', s)
belief     = alpha_T          (renormalised at every step → filtering form)
```

- Renormalising each step is standard filtering and avoids numerical underflow on long
  sequences; it also means `alpha_t` is exactly `P(state_t | obs_1..t)`.
- **Default for roads with no observations:** the `initial` distribution (documented;
  the alternative is the chain's stationary distribution — `initial` chosen for
  simplicity since it is given directly).
- `compute_traffic_beliefs` returns a belief for every road.

**Hand-check (unit test).** One `High` observation → belief ∝
`initial(s)·emission(s,High)` = Low 0.006 / Medium 0.030 / High 0.080, normalised by
0.116. Also tested: empty sequence returns `initial`; six `High`s drive `P(High) > 0.5`.

### Key decisions / possible-wrong flags

- Treating the three report sources as conditionally independent is the modelling
  assumption that makes the product-of-likelihoods valid; if sources secretly copy each
  other (e.g. police repeating citizen reports) this would double-count evidence.
- The HMM belief is the filtered estimate at the **last** observation. Because
  observations are historical (`observation_nr` is a sequence index, not a timestamp),
  we treat "most recent observation" as "current enough" — a simplification.
- Both modules return a belief for *every* road so later stages never special-case
  missing data.

### Verified

`pytest tests/test_bayes.py tests/test_hmm.py` → **14 passed**. Covers: distributions
sum to 1 for all roads in all three scenarios, the two hand-computed values above,
default/edge-case sequences, and that unreported roads fall back to the marginal prior.
(`pytest` was installed into `.venv`.)

---

## Batch 3 — cost model + route search

**Files:** `ers/graph.py`, `ers/search.py`, `tests/test_search.py`.

### `graph.py` — from beliefs to expected travel times

This is the bridge from probabilistic belief to a concrete cost the search can
minimise. Per road:

```
expected_time = (distance / speed) · status_multiplier · traffic_multiplier
```

- **status_multiplier** (from Bayes, conditioned on *not blocked*):
  `(P(Open)·1.0 + P(Congested)·1.5) / (P(Open) + P(Congested))`. We condition on
  not-blocked because blocked roads are removed entirely rather than priced.
- **traffic_multiplier** (from the HMM): `Σ_level P(level)·mult(level)` with
  Low 1.0 / Medium 1.3 / High 1.6. Rationale: historically heavier traffic slows a
  road even when its status is Open — so status and traffic capture two different
  effects and are modelled as independent factors.
- **Blocked handling:** a road with `P(Blocked) > blocked_threshold` (0.5) is dropped
  from `adjacency` and recorded in `blocked`, so both the planner and the validator
  refuse to route over it.
- All parameters live in the `PlanningParams` dataclass (tunable, defaults shown).

Because both multipliers are **≥ 1**, every edge cost is ≥ its free-flow time — the
property the A* heuristic relies on.

### `search.py` — A* (+ Dijkstra oracle)

- `a_star(graph, start, goal)` returns `(path, cost)` or `None`. Standard A* with a
  closed set; the priority queue stores `(f, counter, node)` so ties never compare
  node strings and ordering stays deterministic.
- **Heuristic** `h(n) = euclidean(n, goal) / max_speed`. **Admissible** because any
  real path is at least the straight-line distance (triangle inequality) and every
  edge cost ≥ free-flow time ≥ `dist/max_speed`; it is also **consistent**, so A* is
  optimal without re-expansions. I verified on the data that every road's stored
  `distance ≥ euclidean(from, to)` (min ratio 1.017 across all 104 roads), which the
  triangle-inequality argument needs.
- `dijkstra` is kept only as a test oracle (A* with `h = 0`).

### Key decisions / possible-wrong flags

- Multiplying status and traffic factors is a modelling choice, not derived from the
  data. It is monotone and easy to defend, but a road that is both Congested *and*
  High-traffic gets `1.5 × 1.6 = 2.4×`, which may overstate the combined effect.
- Dropping a road at `P(Blocked) > 0.5` is a hard threshold: a road at 0.49 is used at
  full (non-blocked) cost, at 0.51 it vanishes. The threshold is a parameter so this
  cliff can be tuned/defended.

### Verified

`pytest tests/test_search.py` → **10 passed**. Covers: heuristic ≤ true cost for every
node pair in all three scenarios (admissibility), **A* cost == Dijkstra cost** for every
pair, returned paths are connected and use only real non-blocked roads, `start==goal`
gives cost 0, the believed-blocked road `(N1,N2)` is removed, and a stricter blocked
threshold drops at least as many roads.

---

## Batch 4 — allocation (event-driven CSP) + planner

**Files:** `ers/allocation.py`, `ers/planner.py`, `tests/test_allocation.py`.

### Two layers

**1. Event-driven simulation (`run_simulation`).** Enforces *no look-ahead*. Decision
moments are the times when the decision set can change: incident timestamps and
responder free-up times. We process them in increasing order from a heap; new free-up
times (`arrival + time_required`) are pushed as assignments are made. At each moment we
see only incidents already known and responders already free — never the future.

Responder state is two mutable dicts: `free_at[r]` (next time it can depart) and
`locations[r]` (where it currently is; it stays at the last incident it handled).

**2. CSP at each moment (`solve_event_csp`).**
- *Variables* = currently-open incidents.
- *Domain* of an incident = free responders that (a) have the required capability,
  (b) can route to it, and (c) arrive `≤ deadline` — plus the implicit option of
  leaving it unassigned.
- *Constraint* between variables: a responder serves at most one incident.
- *Search* = recursive backtracking with **MRV** (branch on the incident with the
  fewest remaining candidates first), **shortest-response value ordering**, and
  **forward checking** (a chosen responder is removed from the other incidents'
  candidate sets). Among all feasible complete choices we keep the one maximising
  `Σ priority_weight(served)`, breaking ties by **minimum total response time**.

`priority_weight` = Critical 1000 / High 100 / Medium 10 / Low 1, so a more urgent
incident outweighs any number at the level below (domains per moment are small).

### Planner (`planner.py`)

`plan_scenario(folder)` runs load → Bayes + HMM → graph → simulation and returns a
`PlanResult`. `to_plan_json()` emits the required records (`responder_id`,
`incident_id`, `path`, `time`=departure), ordered by departure time.

### Key decisions / possible-wrong flags

- **Greedy per moment, not globally optimal.** Each decision moment is solved
  independently; the system never delays a free responder in anticipation of a better
  future incident. Documented simplification.
- **`time` = departure time** (the decision moment), as agreed.
- **Skip policy:** an incident with no feasible responder *now* stays open and is
  retried at later moments (e.g. when a busy responder frees); if it is never feasible
  it ends unhandled. With this policy every assignment that *is* made meets its
  deadline (a test asserts this).
- Feasibility uses **expected** arrival, so a served incident could still miss under an
  unlucky road draw.

### What the scenarios show (sanity, not a bug)

- Scenario 1: 4/7 served. Scenario 2: 3/12. Scenario 3: 11/20.
- Scenario 2's low count is legitimate: deadlines are tight (e.g. I1's only capable unit
  arrives 56.2 s vs a 55 s deadline), and responder **R9 is stranded at N1**, which the
  blocked roads cut off from the rest of the graph — so I2/I3 genuinely compete for the
  single reachable police unit and the Critical one wins. Verified by hand with A*.

### Verified

`pytest tests/test_allocation.py` → **9 passed**. Real-scenario invariants (capability
match; `start ≥ timestamp` and `≥ free_at`; path ends at the incident; each incident at
most once; no overlapping jobs per responder; handled/unhandled partition the incidents;
every made assignment meets its deadline) plus a **synthetic CSP** where a naive greedy
would strand a Critical incident but MRV+backtracking serves both, and infeasible
deadline / capability-mismatch cases that correctly yield no assignment.

---

## Batch 5 — outputs + independent validation + entry point

**Files:** `ers/stats.py`, `ers/validator.py`, `run.py`, `tests/test_validator.py`,
`tests/test_scenarios.py`.

### `stats.py` — statistics.txt

Reports incidents handled, responders used, average response time, % deadlines met, and
the unhandled count + list. The metric **definitions are written into the file** so the
output is self-documenting (units in seconds; response time = arrival − timestamp;
average over handled incidents; % deadlines met over handled incidents, with unhandled
listed separately).

### `validator.py` — independent checks

`validate_plan(scenario, plan)` re-derives the beliefs and the blocked set from the
scenario (it does **not** trust the planner's state) and replays the plan in dispatch-time
order, tracking each responder's location and free time. It checks every constraint from
the assignment: path endpoints, every step is a real non-blocked road, capability match,
`start ≥ timestamp` and `≥ free_at`, no overlapping jobs, each incident once, and reports
the deadline verdict per assignment. *Limitation:* arrival is recomputed from the same
expected cost model, so it validates constraint satisfaction and internal consistency,
not a specific real road outcome.

### `run.py` — entry point

`python run.py <scenario_folder> [output_folder]` runs the pipeline, writes `plan.json`
and `statistics.txt` (to the CWD by default), then validates the plan and prints a
summary (exit code 1 if the plan is somehow invalid).

### Bug found and fixed during this batch

The first run flagged **Scenario 3 INVALID**. Cause was in *my own output*, not the
planner: `plan.json`'s departure `time` was rounded to 3 decimals, so a responder
appeared to leave ~0.0004 s before it was actually free, which cascaded into a
stale-location error on its next job. Fix: stop rounding non-integer departure times —
`plan.json` now stores the exact departure, keeping the plan internally consistent.
Clean scenarios still emit integer times. This is a good example of the validator
earning its keep.

### Results (all three scenarios, VALID)

| Scenario | Handled | Responders used | Avg response (s) | Deadlines met | Unhandled |
|---|---|---|---|---|---|
| 1 | 4/7 | 3 | 103.5 | 100% | I1, I6, I7 |
| 2 | 3/12 | 2 | 21.3 | 100% | 9 incidents (tight deadlines + stranded R9) |
| 3 | 11/20 | 6 | 684.1 | 100% | 9 incidents |

100% "deadlines met" is a consequence of the **skip policy**: only feasible assignments
are made, so every assignment that exists meets its deadline. The honest headline number
is therefore *coverage* (handled/total), which the statistics also report.

### Verified

`pytest tests/test_validator.py tests/test_scenarios.py` passes. The validator tests check
**both directions**: generated plans validate for all three scenarios, and each kind of
corruption (capability mismatch, non-existent road, blocked road, duplicate incident,
dispatch before timestamp) is actually rejected. The scenario test runs `run.py` end to
end into a temp dir and re-validates the written `plan.json`.

---

## Batch 6 — packaging: README, requirements, ignore rules

**Files:** `README.md`, `requirements.txt`, `.gitignore` (updated).

- `README.md`: install/run instructions, the concept→module table, the pipeline diagram,
  the tunable parameters, the full assumptions list, and known limitations.
- `requirements.txt`: only `pytest` (the system itself is pure standard library).
- `.gitignore`: also ignores generated `plan.json` / `statistics.txt` / `out/` so build
  artifacts never get committed.

### Final state

- **44 tests pass** (`pytest -q`).
- `python run.py Scenarios/ScenarioN` produces a **VALID** `plan.json` +
  `statistics.txt` for all three scenarios.
- Repository layout: `run.py` entry point, `ers/` package (one module per concept),
  `tests/` suite, `Scenarios/` data, and this notes file.

### What a defence should stress (honest summary)

- The four concepts are real and separable: Bayes and HMM each produce a belief; the graph
  turns beliefs into costs; A* is provably optimal here (admissible+consistent heuristic,
  equal to Dijkstra); the CSP is genuine backtracking with MRV + forward checking.
- The main simplifications: static beliefs, independent multiplicative penalties,
  greedy-per-moment allocation, and expected-time deadlines. Each is documented and
  testable, and the "100% deadlines met" figure is a consequence of the skip policy — so
  coverage is the number to discuss, not that percentage.

---

## Change: serve late instead of skip

**Why.** With the skip policy, `statistics.txt` reported "Deadlines met: 100%"
next to many unhandled incidents (e.g. 9 of 12 in Scenario 2). The 100% was
true by construction (only on-time assignments were ever made) and hid the
real picture. More fundamentally, an emergency whose deadline cannot be met
still needs a responder: skipping it is not acceptable.

**What changed.**
- `allocation.py`: the deadline is now a *soft* constraint. The domain of an
  incident = free responders with the capability and a route; late arrivals
  are allowed and carry a `lateness`. Value ordering: on-time first, then
  shortest route. The CSP score is compared lexicographically:
  (1) priority served, (2) priority served on time, (3) − total lateness,
  (4) − total travel. So the planner first serves as much as it can, then
  makes as much of it as possible on time.
- `stats.py`: "Deadlines met" is now a percentage of **all** incidents; late
  arrivals and any unhandled incidents count as missed.
- Tests: the old "every assignment meets its deadline" and "infeasible
  deadline → unassigned" tests were replaced by "reachable incidents are never
  skipped", "missed deadline is served late" and "on-time responder preferred
  over a late one". 45 tests pass.

**Results (all VALID).**

| Scenario | Handled | Responders used | Avg response (s) | Deadlines met | Unhandled |
|---|---|---|---|---|---|
| 1 | 7/7 | 4 | 109.9 | 57.1% (4/7) | 0 |
| 2 | 12/12 | 9 | 41.2 | 33.3% (4/12) | 0 |
| 3 | 20/20 | 9 | 1613.6 | 40.0% (8/20) | 0 |

**Trade-off.** In Scenario 3 the number of on-time arrivals drops from 11 to 8:
responders sent to already-late incidents are busy when later incidents appear
that they could have reached in time. Without look-ahead the planner cannot
anticipate this. The average response time also rises because late (long)
responses are now included.
