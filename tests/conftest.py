"""Shared pytest fixtures and path setup.

Adds the project root to ``sys.path`` so ``import ers`` works when pytest is run
from anywhere, and exposes the three bundled scenarios as fixtures.
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ers.data_loader import load_scenario  # noqa: E402

SCENARIOS_DIR = os.path.join(ROOT, "Scenarios")


@pytest.fixture
def scenario1():
    return load_scenario(os.path.join(SCENARIOS_DIR, "Scenario1"))


@pytest.fixture
def scenario1_folder():
    return os.path.join(SCENARIOS_DIR, "Scenario1")


@pytest.fixture(params=["Scenario1", "Scenario2", "Scenario3"])
def any_scenario(request):
    return load_scenario(os.path.join(SCENARIOS_DIR, request.param))


@pytest.fixture(params=["Scenario1", "Scenario2", "Scenario3"])
def any_scenario_folder(request):
    return os.path.join(SCENARIOS_DIR, request.param)
