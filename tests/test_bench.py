from pathlib import Path

from probe.bench import load_matrix, matrix_hash, run_matrix

SMOKE = """
version: smoke
seeds: 1
budget: small
primary_policy: P8
baselines: [P1]
policies: [P1, P8]
positive_control: {family: systemlab, regime: R2}
cells:
  - {family: systemlab, regime: R2, expected: tie}
"""


def test_bench_smoke(tmp_path: Path):
    m = tmp_path / "m.yaml"
    m.write_text(SMOKE, encoding="utf-8")
    out = run_matrix(m, tmp_path / "out")
    assert out["meta"]["n_runs"] == 2
    assert (tmp_path / "out" / "regime_map.html").exists()
    assert (tmp_path / "out" / "results.parquet").exists()
    assert out["meta"]["matrix_hash"] == matrix_hash(load_matrix(m))


def test_matrix_v0_is_valid():
    m = load_matrix(Path(__file__).parents[1] / "bench" / "matrix_v0.yaml")
    assert m["positive_control"]["regime"] == "R10"
