# Emergency Response System (ERS)

An AI decision-support system for emergency response after a severe storm. For a
given scenario it estimates uncertain road conditions, assigns responders to
incidents, and plans routes through the city graph — then writes `plan.json` and
`statistics.txt`.

It implements four course concepts, each in its own module:

| Concept | Module | What it does |
|---|---|---|
| **Bayesian reasoning** | `ers/bayes.py` | posterior `P(Status \| weather, reports)` per road |
| **Hidden Markov Model** | `ers/hmm.py` | forward algorithm → belief over current traffic level |
| **Search** | `ers/search.py` | A* over expected travel times (Dijkstra as oracle) |
| **Constraint-based allocation** | `ers/allocation.py` | event-driven simulation + backtracking CSP |

A step-by-step explanation of every design decision is in
[`IMPLEMENTATION_NOTES.md`](IMPLEMENTATION_NOTES.md).

## Requirements

- Python 3.9+ (standard library only for the system itself).
- `pytest` is needed only to run the tests.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # installs pytest
```

## Running

```bash
python run.py <scenario_folder> [output_folder]
```

Examples:

```bash
python run.py Scenarios/Scenario1            # writes to the current directory
python run.py Scenarios/Scenario2 out/s2     # writes to out/s2/
```

This generates:

- **`plan.json`** — a list of assignments, each with `responder_id`,
  `incident_id`, `path` (list of node ids), and `time` (departure time).
- **`statistics.txt`** — incidents handled, responders used, average response
  time, % deadlines met, unhandled incidents, and the definitions used.

The entry point also validates the plan it produced and prints whether it is
valid. All times are in **seconds**, distances in **metres**, speeds in **m/s**.

## Tests

```bash
pytest -q
```

The suite covers each component (a hand-checkable Bayes posterior, HMM beliefs
summing to 1, A* cost equal to Dijkstra, the CSP respecting capabilities and
availability) and runs all three scenarios through the independent validator,
including deliberately corrupted plans to confirm the validator rejects them.

## How it works (pipeline)

```
scenario folder
   ├─ bayes : P(Open/Congested/Blocked) per road from weather + noisy reports
   ├─ hmm   : P(Low/Medium/High) current traffic per road from observations
   ├─ graph : expected time = (distance/speed) · status_mult · traffic_mult;
   │          roads with P(Blocked) > 0.5 are removed
   ├─ allocation : event-driven simulation; at each decision moment a small CSP
   │               (backtracking + MRV + forward checking) assigns free, capable
   │               responders to open incidents, preferring priority then speed
   └─ outputs : plan.json + statistics.txt  (validated independently)
```

## Key parameters (tunable in `ers/graph.py` / `ers/allocation.py`)

- Traffic multipliers: Low 1.0, Medium 1.3, High 1.6.
- Congested multiplier: 1.5 (fixed by the assignment).
- Blocked threshold: drop a road when `P(Blocked) > 0.5`.
- Priority weights: Critical 1000, High 100, Medium 10, Low 1.

## Assumptions

1. Road-condition beliefs are computed once at `t = 0` and treated as static
   during the simulation (reports are available at time 0; traffic observations
   are historical).
2. A responder stays at the incident location after finishing.
3. A responder is occupied from **departure until arrival + time_required**.
4. `"time"` in `plan.json` is the **departure** time.
5. Deadline feasibility is judged on **expected** travel times, so a plan can be
   feasible in expectation yet miss under an unlucky road outcome.
6. **No look-ahead:** an incident exists for the planner only from its timestamp.
7. **Missed-deadline policy: serve late.** The deadline is a soft constraint:
   if no capable responder can arrive in time, the best one is still dispatched
   (on-time options are always preferred). An incident is only left unhandled
   if no capable responder can reach it at all.

## Known limitations

- Allocation is greedy **per decision moment**, not globally optimal over the
  whole incident stream.
- Road status and traffic penalties are combined by multiplying two independent
  factors — a defensible but not derived modelling choice.
- Serving late is myopic: a responder sent to a late incident is busy when a
  later incident (which it could have reached on time) appears. With no
  look-ahead the planner cannot trade these off; in Scenario 3 this lowers the
  on-time count compared to a skip policy (8 vs 11) while serving all 20
  incidents instead of 11.

See `IMPLEMENTATION_NOTES.md` for the full reasoning, per-batch, including a bug
the validator caught during development.
