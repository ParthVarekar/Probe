"""Golden decision-sequence hashes.

Regenerate with: PROBE_REGEN_GOLDEN=1 uv run pytest tests/golden
"""

import json
import os
from pathlib import Path

import pytest

from probe.run import run_world

PATH = Path(__file__).with_name("decision_hashes.json")
CASES = [
    ("microclimate", "R1", 0, "P8"),
    ("microclimate", "R4", 1, "P7"),
    ("microclimate", "R10", 2, "P3"),
    ("systemlab", "R1", 0, "P8"),
    ("systemlab", "R2", 1, "P6"),
]


def _key(c):
    return "/".join(map(str, c))


def _compute():
    return {_key(c): run_world(*c)[0].decision_hash for c in CASES}


def test_golden_decision_hashes():
    got = _compute()
    if os.environ.get("PROBE_REGEN_GOLDEN") == "1" or not PATH.exists():
        PATH.write_text(json.dumps(got, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        pytest.skip("golden hashes regenerated")
    assert got == json.loads(PATH.read_text(encoding="utf-8"))
