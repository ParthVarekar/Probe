"""``probe`` command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from probe import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="probe",
        description="PROBE: verifiable adaptive experimentation (pre-alpha).",
    )
    parser.add_argument("--version", action="version", version=f"probe {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    run = sub.add_parser("run", help="run one policy on one simulated world")
    run.add_argument("--family", default="microclimate", choices=["microclimate", "systemlab"])
    run.add_argument("--regime", default="R1")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--policy", default="P8")
    run.add_argument("--budget", default="medium", choices=["small", "medium", "large"])
    run.add_argument("--json", type=Path, help="write the full run record as JSON")

    bench = sub.add_parser("bench", help="run a benchmark regime matrix")
    bench.add_argument("--matrix", type=Path, default=Path("bench/matrix_v0.yaml"))
    bench.add_argument("--seeds", type=int, default=None, help="override seeds per cell")
    bench.add_argument("--workers", type=int, default=1)
    bench.add_argument("--out", type=Path, default=Path("results/bench"))
    return parser


def _cmd_run(args: argparse.Namespace) -> int:
    from probe.determinism import canonical_json, to_jsonable
    from probe.run import run_world

    record, metrics = run_world(args.family, args.regime, args.seed, args.policy, args.budget)
    print(f"SIMULATED world {args.family}/{args.regime} seed {args.seed}, policy {args.policy}")
    print(f"  stop: {record.stop['reason']}  verdict: {metrics['verdict']}")
    post = ", ".join(f"{h}={w:.3f}" for h, w in record.final_weights.items())
    print(f"  posterior: {post}")
    print(f"  steps: {len(record.steps)}  cost: {record.spent:.1f}  n_obs: {record.n_obs}")
    print(f"  decision hash: {record.decision_hash}")
    print(
        f"  evaluator: correct={metrics['correct']} "
        f"confidently_wrong={metrics['confidently_wrong']}"
    )
    if args.json:
        args.json.write_bytes(canonical_json(to_jsonable(record)))
        print(f"  record written to {args.json}")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from probe.bench import run_matrix

    summary = run_matrix(args.matrix, args.out, seeds=args.seeds, workers=args.workers)
    print(json.dumps(summary, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "bench":
        return _cmd_bench(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
