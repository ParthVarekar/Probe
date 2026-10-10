"""``probe`` command-line entry point."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from probe import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="probe",
        description="PROBE: verifiable adaptive experimentation (pre-alpha).",
    )
    parser.add_argument("--version", action="version", version=f"probe {__version__}")
    parser.add_subparsers(dest="command", metavar="<command>")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
