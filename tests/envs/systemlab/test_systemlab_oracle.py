"""Reset-capable Mealy oracle (automata-learning baseline P11) and an AALpy L* smoke run."""

import itertools
import random

import numpy as np
import pytest

from probe.envs.systemlab import FAMILY, MealyOracle, aalpy_sul
from probe.envs.systemlab.services import Cache, TokenBucket
from probe.truth import EvaluatorKey

KEY = EvaluatorKey("oracle tests")


def test_cache_oracle_matches_the_planted_cache():
    world = FAMILY.make_world("R11", 4)
    p = world.truth.open(KEY)["payload"]["service"]["params"]
    oracle = world.env.reset_oracle()
    rng = np.random.default_rng(0)
    for _ in range(30):
        word = "".join(rng.choice(list(oracle.alphabet), size=int(rng.integers(1, 12))))
        ref = Cache(p["policy"], p["size"])
        assert oracle.query(word) == ["hit" if ref.access(k) else "miss" for k in word]
    assert oracle.n_resets == 30
    assert oracle.cost == oracle.n_steps + oracle.n_resets
    assert "policy" not in repr(oracle) and "size" not in repr(oracle)


def test_rate_oracle_semantics():
    oracle = MealyOracle(lambda: TokenBucket(2, 3), MealyOracle.RATE_ALPHABET, reset_cost=5.0)
    out = oracle.query(["req", "req", "req", "tick", "tick", "req", "tick", "req"])
    assert out == ["ok", "ok", "rej", "-", "-", "rej", "-", "ok"]
    assert oracle.query(["req", "req"]) == ["ok", "ok"]  # reset restores a full bucket
    assert oracle.cost == 10 + 2 * 5.0
    with pytest.raises(ValueError):
        oracle.step("burst")


def _all_words(alphabet, max_len):
    for n in range(1, max_len + 1):
        yield from itertools.product(alphabet, repeat=n)


def test_aalpy_lstar_learns_the_r11_machine():
    pytest.importorskip("aalpy")
    from aalpy.learning_algs import run_Lstar
    from aalpy.oracles import StatePrefixEqOracle

    world = FAMILY.make_world("R11", 0)  # an MRU cache (finite-state, outside the set)
    oracle = world.env.reset_oracle()
    sul = aalpy_sul(oracle)
    alphabet = list(oracle.alphabet)
    saved = random.getstate()
    try:
        random.seed(1234)  # AALpy's equivalence oracles use the global `random` module
        eq = StatePrefixEqOracle(alphabet, sul, walks_per_state=10, walk_len=8)
        model = run_Lstar(alphabet, sul, eq, automaton_type="mealy", print_level=0)
    finally:
        random.setstate(saved)
    assert oracle.n_resets > 0 and oracle.cost > 0
    check = world.env.reset_oracle()
    for word in _all_words(alphabet, 5):
        model.reset_to_initial()
        assert [model.step(x) for x in word] == check.query(word)
