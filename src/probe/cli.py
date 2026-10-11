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

    rec = sub.add_parser("record", help="run and record an experiment in a ledger workspace")
    rec.add_argument("--workspace", type=Path, default=Path("probe_workspace"))
    rec.add_argument("--family", default="microclimate", choices=["microclimate", "systemlab"])
    rec.add_argument("--regime", default="R1")
    rec.add_argument("--seed", type=int, default=0)
    rec.add_argument("--policy", default="P8")
    rec.add_argument("--budget", default="medium", choices=["small", "medium", "large"])

    exp = sub.add_parser("export", help="export a recorded experiment as a bundle zip")
    exp.add_argument("experiment_id")
    exp.add_argument("--workspace", type=Path, default=Path("probe_workspace"))
    exp.add_argument("--out", type=Path, default=None)

    ver = sub.add_parser("verify", help="verify a bundle: hash chain, ordering, replay")
    ver.add_argument("bundle", type=Path)
    ver.add_argument("--no-replay", action="store_true")

    rep = sub.add_parser("report", help="render the HTML report of a bundle (verifies it first)")
    rep.add_argument("bundle", type=Path)
    rep.add_argument("--out", type=Path, default=None)
    return parser


def _cmd_report(args: argparse.Namespace) -> int:
    import zipfile

    from probe.bundle import verify_bundle
    from probe.report import render_report

    ver = verify_bundle(args.bundle)
    with zipfile.ZipFile(args.bundle) as z:
        events = [json.loads(x) for x in z.read("ledger.jsonl").decode().splitlines() if x]
        manifest = json.loads(z.read("manifest.json"))
        last = [
            e
            for e in events
            if e["experiment_id"] == manifest["experiment_id"] and e["type"] == "run_record"
        ]
        record = json.loads(z.read(f"blobs/{last[-1]['payload_hash']}"))
    run = manifest["run"]
    label = f"{run['family']}/{run['regime']}"
    out = args.out or args.bundle.with_suffix(".html")
    out.write_text(render_report(record, world_label=label, verification=ver), encoding="utf-8")
    print(out)
    return 0


def _cmd_record(args: argparse.Namespace) -> int:
    from probe.bundle import record_run

    exp, m = record_run(
        args.workspace, args.family, args.regime, args.seed, args.policy, args.budget
    )
    print(f"recorded {exp}: verdict {m['verdict']} (SIMULATED)")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    from probe.bundle import export_bundle

    out = args.out or Path(f"{args.experiment_id}.probe.zip")
    print(export_bundle(args.workspace, args.experiment_id, out))
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    from probe.bundle import verify_bundle

    rep = verify_bundle(args.bundle, replay=not args.no_replay)
    print(json.dumps(rep, indent=2))
    return 0 if rep["ok"] else 1


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
    if args.command == "record":
        return _cmd_record(args)
    if args.command == "export":
        return _cmd_export(args)
    if args.command == "verify":
        return _cmd_verify(args)
    if args.command == "report":
        return _cmd_report(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
