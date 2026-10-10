import subprocess
import sys

import probe
from probe.cli import main


def test_version_string():
    assert probe.__version__.startswith("0.1.0")


def test_cli_help_returns_zero(capsys):
    assert main([]) == 0
    assert "probe" in capsys.readouterr().out


def test_cli_version_subprocess():
    out = subprocess.run(
        [sys.executable, "-m", "probe.cli", "--version"], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == f"probe {probe.__version__}"
