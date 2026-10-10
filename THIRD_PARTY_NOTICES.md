# Third-party notices

PROBE is licensed under the Apache License 2.0 (see `LICENSE`). It depends on
the following third-party packages, installed from PyPI and not vendored. Each
is used under its own licence.

| Package | Use | Licence |
|---|---|---|
| numpy | Linear algebra, seeded RNG (PCG64) | BSD-3-Clause |
| scipy | Special functions, quadrature, statistics | BSD-3-Clause |
| pydantic | Hypothesis IR and protocol schemas | MIT |
| PyYAML | Regime-matrix files | MIT |
| pvlib | Solar position and clear-sky irradiance in the microclimate simulator | BSD-3-Clause |
| pandas (via pvlib) | Time-series inputs to pvlib | BSD-3-Clause |
| pyarrow | Parquet benchmark results | Apache-2.0 |

Development and test only (not shipped):

| Package | Use | Licence |
|---|---|---|
| pytest, pytest-xdist | Test runner | MIT |
| hypothesis | Property-based tests | MPL-2.0 |
| dynesty | Independent evidence oracle in tests | MIT |
| AALpy | Automata-learning baseline for the system lab | MIT |
| pyro-ppl (optional `oracle` group) | Independent EIG oracle in Linux CI | Apache-2.0 |
| ruff, mypy | Lint and type checks | MIT |

No GPL, AGPL or EPL code is included in or linked into the core (handover §12 Part M).

When code adapted from a public-domain or permissively licensed project is added
(for example NIST `optbayesexpt` patterns), its notice will be recorded here.
