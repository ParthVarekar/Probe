"""Reset-capable Mealy oracle (automata-learning baseline P11), the exact equivalence
check used to score it, and AALpy L* on every R11 variant within the regime budget."""

import itertools
import random

import numpy as np
import pytest

from probe.envs.systemlab import (
    FAMILY,
    R11_VARIANTS,
    REGIME_IDS,
    MealyOracle,
    ServiceSpec,
    aalpy_sul,
    build_service,
    mealy_counterexample,
    reachable_states,
)
from probe.envs.systemlab.services import Cache, TokenBucket
from probe.truth import EvaluatorKey

KEY = EvaluatorKey("oracle tests")
ALPHABET = ("a", "b", "c", "d")


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


@pytest.mark.parametrize("regime", REGIME_IDS)
def test_service_spec_round_trips_through_the_sealed_payload(regime):
    for seed in range(6):
        d = FAMILY.make_world(regime, seed).truth.open(KEY)["payload"]["service"]
        spec = ServiceSpec.from_dict(d)
        assert spec.as_dict() == d
        build_service(spec)


@pytest.mark.parametrize("policy", ["lru", "fifo", "mru", "clock", "lfu"])
@pytest.mark.parametrize("size", [1, 2, 3])
def test_state_key_is_a_sound_behavioural_state(policy, size):
    """Caches reached by different histories with equal keys answer every suffix alike;
    the key ignores absolute counters (finitely many keys for all but LFU)."""
    rng = np.random.default_rng(size)
    by_key: dict = {}
    for _ in range(300):
        c = Cache(policy, size)
        c.run("".join(rng.choice(list("abcd"), size=int(rng.integers(0, 10)))))
        by_key.setdefault(c.state_key(), []).append(c)
    shared = [cs for cs in by_key.values() if len(cs) > 1]
    assert shared
    for caches in shared:
        for _ in range(5):
            suffix = "".join(rng.choice(list("abcd"), size=12))
            outs = {tuple(c.clone().access(k) for k in suffix) for c in caches}
            assert len(outs) == 1
    if policy != "lfu":
        assert reachable_states(lambda: Cache(policy, size), ALPHABET) >= len(by_key)


def test_reachable_state_counts():
    counts = {v: reachable_states(lambda v=v: Cache(*v), ALPHABET) for v in R11_VARIANTS}
    assert counts == {("mru", 2): 17, ("mru", 3): 29, ("clock", 2): 105}
    assert reachable_states(lambda: Cache("lru", 1), ALPHABET) == 5
    with pytest.raises(RuntimeError):
        reachable_states(lambda: Cache("lfu", 2), ALPHABET, limit=500)  # unbounded counts
    with pytest.raises(TypeError):
        reachable_states(lambda: TokenBucket(2, 1), MealyOracle.RATE_ALPHABET)


class _State:
    def __init__(self) -> None:
        self.transitions: dict = {}
        self.output_fun: dict = {}


class _Machine:
    def __init__(self, initial: _State) -> None:
        self.initial_state = initial


def _true_machine(policy: str, size: int) -> _Machine:
    """The planted cache unrolled into an explicit Mealy machine (AALpy's shape)."""
    c0 = Cache(policy, size)
    states = {c0.state_key(): _State()}
    queue = [c0]
    while queue:
        c = queue.pop()
        s = states[c.state_key()]
        for a in ALPHABET:
            d = c.clone()
            s.output_fun[a] = "hit" if d.access(a) else "miss"
            k = d.state_key()
            if k not in states:
                states[k] = _State()
                queue.append(d)
            s.transitions[a] = states[k]
    return _Machine(states[c0.state_key()])


def _outputs(cache: Cache, word) -> list[str]:
    return ["hit" if cache.access(a) else "miss" for a in word]


def _run(machine: _Machine, word) -> list[str]:
    s, out = machine.initial_state, []
    for a in word:
        out.append(s.output_fun[a])
        s = s.transitions[a]
    return out


@pytest.mark.parametrize(("policy", "size"), [*R11_VARIANTS, ("lru", 2), ("fifo", 3)])
def test_mealy_counterexample_is_exact(policy, size):
    machine = _true_machine(policy, size)
    assert mealy_counterexample(machine, lambda: Cache(policy, size), ALPHABET) is None
    for other in [("lru", 2), ("fifo", 2), ("mru", 2), ("clock", 2), ("lru", 3)]:
        if other == (policy, size):
            continue
        word = mealy_counterexample(machine, lambda o=other: Cache(*o), ALPHABET)
        assert word is not None
        got, want = _run(machine, word), _outputs(Cache(*other), word)
        assert got[:-1] == want[:-1] and got[-1] != want[-1]  # shortest: differs only at the end
        for n in range(1, len(word)):  # no shorter word distinguishes them
            for w in itertools.product(ALPHABET, repeat=n):
                assert _run(machine, w) == _outputs(Cache(*other), w)


def _learn(world, rand_seed: int):
    pytest.importorskip("aalpy")
    from aalpy.learning_algs import run_Lstar
    from aalpy.oracles import StatePrefixEqOracle

    oracle = world.env.reset_oracle()
    sul = aalpy_sul(oracle)
    alphabet = list(oracle.alphabet)
    saved = random.getstate()
    try:
        random.seed(rand_seed)  # AALpy's equivalence oracles use the global `random` module
        eq = StatePrefixEqOracle(alphabet, sul, walks_per_state=10, walk_len=8)
        model = run_Lstar(alphabet, sul, eq, automaton_type="mealy", print_level=0)
    finally:
        random.setstate(saved)
    return model, oracle


@pytest.mark.parametrize("seed", [0, 1, 2])  # one block: every R11 variant once
@pytest.mark.parametrize("rand_seed", [7, 1234])
def test_aalpy_lstar_learns_every_r11_service_exactly_within_budget(seed, rand_seed):
    """The pre-registered R11 expectation, checked: reference L* learns an exactly
    equivalent Mealy machine at an oracle cost inside the regime's large budget (and
    inside the budget tier the regime notes assign to the variant)."""
    world = FAMILY.make_world("R11", seed)
    spec = ServiceSpec.from_dict(world.truth.open(KEY)["payload"]["service"])
    variant = (spec.params["policy"], spec.params["size"])
    model, oracle = _learn(world, rand_seed)
    assert mealy_counterexample(model, lambda: build_service(spec), ALPHABET) is None
    budgets = FAMILY.regimes()["R11"].budgets
    tier = {("mru", 2): "small", ("mru", 3): "medium", ("clock", 2): "large"}[variant]
    assert oracle.n_resets > 0
    assert oracle.cost <= budgets[tier]
    assert len(model.states) <= reachable_states(lambda: build_service(spec), ALPHABET)
