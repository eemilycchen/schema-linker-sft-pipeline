# try/ — local debugging sandbox

Notebooks and helpers here are for **experimentation and debugging**. Production training scripts should stay at the repo root.

## Layout

```
schema-linker-sft-pipeline/   ← clone this whole repo
├── train.json
├── validation.json
├── schemas/
└── try/                      ← you are here
    ├── paths.py              ← single source of truth for paths
    ├── schema_formatting.py
    ├── schema_metrics.py
    ├── logs/                 ← RapidFire experiment dirs (gitignored)
    └── *.ipynb
```

## Setup (every teammate)

1. Clone the **full** repository (not only `try/`).
2. Create/activate your conda env (`cse234` on DSMLP) and install deps (see main `README.md`).
3. Start RapidFire if you use the UI: `rapidfireai start`
4. Open a notebook under `try/` and run cells **top to bottom**.

## Paths — do not hardcode `/home/...`

Always bootstrap in the first code cell:

```python
from paths import setup, require_release_data, TRAIN_JSON, VAL_JSON, LOGS_DIR, REPO_ROOT

setup(chdir_try=True)   # adds try/ to sys.path; cwd → try/ so ./logs works
require_release_data()  # checks train.json, validation.json, schemas/
```

Then load data with:

```python
import json
with open(TRAIN_JSON) as f:
    raw_train = json.load(f)
```

**Why `Path("~")` failed:** `Path("~/foo")` does not expand `~` unless you call `.expanduser()`. Prefer `paths.py` instead of manual home paths.

**Why `Path(".")` failed:** Jupyter’s cwd is not always `try/` or the repo root. `setup(chdir_try=True)` fixes that.

## Logs and RapidFire DB

| What | Where |
|------|--------|
| Experiment artifacts (`datasets.dill`, `runs/`) | `try/logs/<experiment-name>/` |
| RapidFire UI logs | `$RF_HOME/logs/` (default `~/rapidfireai/logs/`) |
| Experiment registry (SQLite) | `~/rapidfireai/db/rapidfire_fit.db` |

To reset debug experiment names:

```bash
rm -rf try/logs/proj2-debug-8config-*
rm -rf ~/rapidfireai/logs/proj2-debug-8config-*

python -c "
import sqlite3
con = sqlite3.connect('$HOME/rapidfireai/db/rapidfire_fit.db')
con.execute(\"DELETE FROM experiments WHERE experiment_name LIKE 'proj2-debug-8config%'\")
con.commit(); con.close()
"
```

If `rm` says `.nfs... Device or resource busy`, restart the notebook kernel first.

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `proj2-debug-8config.ipynb` | 8-config sweep with `RUN_IDS` / `DEBUG` knobs |
| `proj2-scratch.ipynb` | Full project notebook (work in progress) |
| `proj2-preflight.ipynb` | Quick data / import checks |

## KeyboardInterrupt during `run_fit`

Interrupting the cell leaves RapidFire polling in a sleep loop (`time.sleep(1)`). That is normal. Call `experiment.end()`, restart the kernel, then re-run from the experiment cell.
