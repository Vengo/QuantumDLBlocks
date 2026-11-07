"""Verification Runner for `EXPERIMENT_PLAN_078.md` (`V0..V4` Orthogonal Vacuum Subspace & 4-Mode 512D Cholesky).

Delegates to `scripts/vacuum_subspace_cholesky_bench.py`.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "scripts"))

from vacuum_subspace_cholesky_bench import main  # noqa: E402

if __name__ == "__main__":
    main()
