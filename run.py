#!/usr/bin/env python3
"""Entry point: plan one scenario and write plan.json + statistics.txt.

Usage:
    python run.py <scenario_folder> [output_folder]

If ``output_folder`` is omitted the two output files are written to the current
working directory. The plan is also validated and a short summary is printed.
"""

from __future__ import annotations

import json
import os
import sys

from ers.planner import plan_scenario
from ers.stats import build_statistics
from ers.validator import validate_plan


def main(argv: list) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2

    scenario_folder = argv[1]
    output_folder = argv[2] if len(argv) > 2 else os.getcwd()

    if not os.path.isdir(scenario_folder):
        print(f"error: scenario folder not found: {scenario_folder}")
        return 2
    os.makedirs(output_folder, exist_ok=True)

    # 1. Plan: beliefs -> graph -> event-driven CSP allocation.
    result = plan_scenario(scenario_folder)
    plan = result.to_plan_json()

    # 2. Write plan.json.
    plan_path = os.path.join(output_folder, "plan.json")
    with open(plan_path, "w", encoding="utf-8") as fh:
        json.dump(plan, fh, indent=2)
        fh.write("\n")

    # 3. Write statistics.txt.
    stats_text = build_statistics(result)
    stats_path = os.path.join(output_folder, "statistics.txt")
    with open(stats_path, "w", encoding="utf-8") as fh:
        fh.write(stats_text)
        if not stats_text.endswith("\n"):
            fh.write("\n")

    # 4. Validate the plan we just produced and report.
    report = validate_plan(result.scenario, plan)

    print(f"Scenario: {result.scenario.name}")
    print(f"  wrote {plan_path}")
    print(f"  wrote {stats_path}")
    print(f"  assignments: {len(plan)}; unhandled: {len(result.unhandled_incident_ids)}")
    print(f"  plan is {'VALID' if report.valid else 'INVALID'}")
    if not report.valid:
        print(report.summary())
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
