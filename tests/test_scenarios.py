"""End-to-end tests: run the entry point on every scenario and validate output."""

import json
import os

import run as entrypoint
from ers.data_loader import load_scenario
from ers.validator import validate_plan


def test_run_py_writes_valid_outputs(any_scenario_folder, tmp_path):
    out = str(tmp_path)
    code = entrypoint.main(["run.py", any_scenario_folder, out])
    assert code == 0  # entry point returns 0 only when the plan validates

    plan_path = os.path.join(out, "plan.json")
    stats_path = os.path.join(out, "statistics.txt")
    assert os.path.exists(plan_path)
    assert os.path.exists(stats_path)

    # plan.json parses and matches the required schema.
    with open(plan_path, encoding="utf-8") as fh:
        plan = json.load(fh)
    assert isinstance(plan, list)
    for entry in plan:
        assert set(entry) >= {"responder_id", "incident_id", "path", "time"}
        assert isinstance(entry["path"], list) and entry["path"]

    # Re-validate the written plan independently.
    scenario = load_scenario(any_scenario_folder)
    assert validate_plan(scenario, plan).valid

    # statistics.txt reports the required metrics and their definitions.
    stats = open(stats_path, encoding="utf-8").read()
    for needle in (
        "incidents handled",
        "responders used",
        "Average response time",
        "Deadlines met",
        "Definitions:",
    ):
        assert needle in stats
