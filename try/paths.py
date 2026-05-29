"""Portable paths for notebooks and scripts under try/.

Layout assumed:
  schema-linker-sft-pipeline/     <- REPO_ROOT (train.json, schemas/, …)
    try/                          <- TRY_DIR (notebooks, helpers, local logs)
      paths.py
      proj2-debug-8config.ipynb
      logs/                       <- experiment artifacts (gitignored)

Teammates: clone the full repo; run notebooks from try/ or any cwd — call setup() first.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

TRY_DIR = Path(__file__).resolve().parent
REPO_ROOT = TRY_DIR.parent

# Release packet at repo root (shared across the team)
TRAIN_JSON = REPO_ROOT / "train.json"
VAL_JSON = REPO_ROOT / "validation.json"
VAL_INPUT_JSON = REPO_ROOT / "validation_input.json"
SCHEMAS_DIR = REPO_ROOT / "schemas"

# RapidFire / notebook outputs kept inside try/ (not repo root)
LOGS_DIR = TRY_DIR / "logs"

# RapidFire global state (same on every machine; override with env RF_HOME)
RF_HOME = Path(os.path.expanduser(os.environ.get("RF_HOME", "~/rapidfireai")))
RF_DB = RF_HOME / "db" / "rapidfire_fit.db"


def setup(*, chdir_try: bool = True) -> dict[str, Path]:
    """Import helpers from try/ and optionally chdir so ./logs → try/logs."""
    try_str = str(TRY_DIR)
    if try_str not in sys.path:
        sys.path.insert(0, try_str)
    if chdir_try:
        os.chdir(TRY_DIR)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "TRY_DIR": TRY_DIR,
        "REPO_ROOT": REPO_ROOT,
        "LOGS_DIR": LOGS_DIR,
        "SCHEMAS_DIR": SCHEMAS_DIR,
    }


def require_release_data() -> None:
    """Fail fast with a clear message if the release packet is missing."""
    required = [
        ("train.json", TRAIN_JSON),
        ("validation.json", VAL_JSON),
        ("schemas/", SCHEMAS_DIR),
    ]
    missing = [label for label, path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Release packet not found. Clone the full repo (not just try/) and ensure:\n"
            + "\n".join(f"  {REPO_ROOT / m}" for m in missing)
            + f"\n\nCurrent REPO_ROOT={REPO_ROOT}"
        )


def print_paths() -> None:
    """Human-readable path check for notebooks."""
    print(f"REPO_ROOT   = {REPO_ROOT}")
    print(f"TRY_DIR     = {TRY_DIR}")
    print(f"SCHEMAS_DIR = {SCHEMAS_DIR}  ({len(list(SCHEMAS_DIR.glob('*.json')))} json files)"
          if SCHEMAS_DIR.is_dir() else f"SCHEMAS_DIR = {SCHEMAS_DIR}  (MISSING)")
    print(f"LOGS_DIR    = {LOGS_DIR}")
    print(f"RF_HOME     = {RF_HOME}")
    print(f"cwd         = {Path.cwd()}")
