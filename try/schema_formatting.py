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

# before with 0.2533
"""
before with 0.2533 
SYSTEM_PROMPT = (
      "You are a schema-linking assistant. Given a database schema and a question, "
      "output ONLY a JSON object where the keys are table names and the values are lists of column names.\n\n"
      "Example Output Format:\n" 
      "{\n"
      '  "singer": ["singer_id", "name", "birth_year"],\n'
      '  "concert": []\n'
      "}\n\n"
      "If a table is referenced but no specific columns are needed, map it to an empty list []. "
      "Do not output any explanation, markdown, or extra text — only the JSON object."
  )
New one: 0.18
SYSTEM_PROMPT = (
    "You are a schema-linking assistant. Given a database schema and a question, "
    "identify which tables and columns from the schema are needed to answer the question.\n\n"
    "Rules:\n"
    "1. Use ONLY table names that appear exactly in the provided schema — do not invent or paraphrase table names.\n"
    "2. Only include tables that are actually needed to answer the question — do not list all tables.\n"
    "3. Column values must always be a JSON array of strings, even if empty: [] not null or a string.\n"
    "4. Output ONLY the JSON object — no explanation, no markdown.\n\n"
    "Example: for schema `singer(singer_id, name)  concert(concert_id, theme)` and question "
    "'What are the names of singers?', output:\n"
    '{"singer": ["name"]}'
)
"""
SYSTEM_PROMPT = (
      "You are a schema-linking assistant. Given a database schema and a question, "
      "output ONLY a JSON object where the keys are table names and the values are lists of column names.\n\n"
      "Example Output Format:\n" 
      "{\n"
      '  "singer": ["singer_id", "name", "birth_year"],\n'
      '  "concert": []\n'
      "}\n\n"
      "If a table is referenced but no specific columns are needed, map it to an empty list []. "
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
def _truncate_schema_to_budget(schema_s: str, question: str, tokenizer, max_length: int = 2048) -> str:
    """Trim schema lines so the full prompt fits within max_length tokens."""
    lines = schema_s.split("\n")
    overhead_msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Database schema:\n\n\nQuestion: {question}"},
    ]
    overhead = len(tokenizer.encode(
        tokenizer.apply_chat_template(overhead_msgs, tokenize=False, add_generation_prompt=True)
    ))
    budget = max_length - overhead - 64  # 64 buffer for completion tokens
    kept, used = [], 0
    for line in lines:
        line_toks = len(tokenizer.encode(line))
        if used + line_toks > budget:
            break
        kept.append(line)
        used += line_toks
    return "\n".join(kept) if kept else "\n".join(lines[:5])


def format_row(row: dict, schema_style: str = "minimal", tokenizer=None, max_length: int = 2048) -> dict:
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
    if tokenizer is not None:
        schema_s = _truncate_schema_to_budget(schema_s, row["question"], tokenizer, max_length)
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


def make_formatting_func(schema_style: str = "minimal", tokenizer=None, max_length: int = 2048) -> Callable[[dict], dict]:
    """Picklable formatter for RFModelConfig.formatting_func."""
    return partial(format_row, schema_style=schema_style, tokenizer=tokenizer, max_length=max_length)


# Convenience aliases
format_minimal = partial(format_row, schema_style="minimal")
format_full = partial(format_row, schema_style="full")
