"""Schema-linking prompt formatting for RapidFire / TRL SFT.

Import from notebooks and workers — no side effects at import time.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Callable

# Schemas live at repo root (../schemas), not under try/
_DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCHEMAS_DIR = _DEFAULT_REPO_ROOT / "schemas"

_schema_cache: dict[str, dict] = {}


def load_schema(db_id: str, schemas_dir: Path | str = DEFAULT_SCHEMAS_DIR) -> dict:
    """Load Spider-format schema JSON for db_id (spaces → underscores in filename)."""
    schemas_dir = Path(schemas_dir)
    if db_id not in _schema_cache:
        fname = db_id.replace(" ", "_") + ".json"
        with open(schemas_dir / fname) as f:
            _schema_cache[db_id] = json.load(f)
    return _schema_cache[db_id]


def serialize_schema_minimal(schema: dict) -> str:
    """Table names + column names only."""
    lines = []
    table_names = schema["table_names_original"]
    column_names = schema["column_names_original"]
    for t_idx, tname in enumerate(table_names):
        cols = [c for (ti, c) in column_names if ti == t_idx and ti != -1]
        lines.append(f"{tname}({', '.join(cols)})")
    return "\n".join(lines)


def serialize_schema_full(schema: dict) -> str:
    """Table/column names plus types, PK, and FK hints."""
    table_names = schema["table_names_original"]
    column_names = schema["column_names_original"]
    column_types = schema["column_types"]
    primary_keys = set(schema["primary_keys"])
    foreign_keys = schema.get("foreign_keys", [])

    fk_lookup: dict[int, str] = {}
    for src, tgt in foreign_keys:
        tgt_table = table_names[column_names[tgt][0]]
        tgt_col = column_names[tgt][1]
        fk_lookup[src] = f"{tgt_table}.{tgt_col}"

    lines = []
    for t_idx, tname in enumerate(table_names):
        col_parts = []
        for c_idx, (ti, cname) in enumerate(column_names):
            if ti != t_idx:
                continue
            ctype = column_types[c_idx] if c_idx < len(column_types) else "text"
            flags = []
            if c_idx in primary_keys:
                flags.append("PK")
            if c_idx in fk_lookup:
                flags.append(f"FK->{fk_lookup[c_idx]}")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            col_parts.append(f"{cname} {ctype}{flag_str}")
        lines.append(f"{tname}(\n  " + ",\n  ".join(col_parts) + "\n)")
    return "\n".join(lines)


SYSTEM_PROMPT = (
    "You are a schema-linking assistant. Given a natural-language question and a database schema, "
    "output ONLY a JSON object mapping each referenced table name to the list of referenced column names. "
    "If a table is referenced but no specific columns are needed (e.g. SELECT COUNT(*)), map it to []. "
    "Do not output any explanation, markdown, or extra text — only the JSON object."
)


def sort_schema_links(schema_links: dict | None) -> dict:
    """Sort tables/columns; tolerate None column lists."""
    if schema_links is None:
        raise TypeError("schema_links is None — row has no gold labels (wrong dataset or double-format?)")
    if not isinstance(schema_links, dict):
        raise TypeError(f"schema_links must be dict, got {type(schema_links)}")
    out = {}
    for table, cols in sorted(schema_links.items()):
        out[table] = sorted(cols) if cols is not None else []
    return out


# fixing the format issues by preventing double format
def format_row(row: dict, schema_style: str = "minimal") -> dict:
    """
    Convert one train/val example to {prompt, completion} chat format.

    Idempotent: if row is already formatted (has prompt + completion), returns unchanged.
    Safe for RapidFire calling map(formatting_func) more than once.
    """
    if "prompt" in row and "completion" in row:
        return {"prompt": row["prompt"], "completion": row["completion"]}

    links = row.get("schema_links")
    if links is None:
        raise ValueError(
            "schema_links missing on row. "
            f"keys={list(row.keys())} question_id={row.get('question_id')}. "
            "Use validation.json (not validation_input.json) for training eval."
        )

    serialize = serialize_schema_minimal if schema_style == "minimal" else serialize_schema_full
    schema_s = serialize(load_schema(row["db_id"]))
    user_msg = f"Database schema:\n{schema_s}\n\nQuestion: {row['question']}"
    target = json.dumps(sort_schema_links(links), ensure_ascii=False)

    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "completion": [
            {"role": "assistant", "content": target},
        ],
    }


def make_formatting_func(schema_style: str = "minimal") -> Callable[[dict], dict]:
    """Picklable formatter for RFModelConfig.formatting_func."""
    return partial(format_row, schema_style=schema_style)


# Convenience aliases
format_minimal = partial(format_row, schema_style="minimal")
format_full = partial(format_row, schema_style="full")
