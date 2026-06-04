#!/usr/bin/env python3
"""
augment_data.py — Rule-based SQL-to-question augmentation.

Generates 200 new training examples by transforming existing ones using:
  1. Question-stem synonym swaps ("Show" -> "List", "How many" -> "Count")
  2. Inline phrase swaps ("for each" <-> "per", "highest" <-> "maximum")
  3. SQL-structure templates (GROUP BY, MAX, MIN, JOIN, etc.)
  4. Prefix injection as a final fallback

Output: train_augmented.json (200 new examples, same format as train.json)
"""

import json
import re
import random
from pathlib import Path
from collections import Counter

SCHEMAS_DIR = Path(__file__).parent / "schemas"
TRAIN_JSON  = Path(__file__).parent / "train.json"
OUT_JSON    = Path(__file__).parent / "train_augmented.json"
TARGET_N    = 200

# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

_schema_cache: dict = {}

def load_schema(db_id: str) -> dict:
    if db_id not in _schema_cache:
        fname = db_id.replace(" ", "_") + ".json"
        with open(SCHEMAS_DIR / fname) as f:
            _schema_cache[db_id] = json.load(f)
    return _schema_cache[db_id]


# ---------------------------------------------------------------------------
# SQL analysis
# ---------------------------------------------------------------------------

def analyze_sql(sql: str) -> dict:
    u = sql.upper()
    return {
        "has_join":     "JOIN"     in u,
        "has_group_by": "GROUP BY" in u,
        "has_order_by": "ORDER BY" in u,
        "has_distinct": "DISTINCT" in u,
        "has_count":    "COUNT("   in u,
        "has_sum":      "SUM("     in u,
        "has_avg":      "AVG("     in u,
        "has_max":      "MAX("     in u,
        "has_min":      "MIN("     in u,
        "has_where":    "WHERE"    in u,
        "has_having":   "HAVING"   in u,
        "has_top":      ("TOP " in u) or ("LIMIT " in u),
        "is_desc":      "DESC"     in u,
    }


# ---------------------------------------------------------------------------
# Stem transformations (replace the opening phrase of the question)
# ---------------------------------------------------------------------------

STEM_RULES = [
    # (regex to match at start of lowercased question, list of replacement prefixes)
    (r"^show\s+me\s+a\s+list\s+of\s+",  ["List all ", "Find all ", "Return all ", "Display all "]),
    (r"^show\s+me\s+the\s+",            ["Find the ", "List the ", "Return the ", "Display the "]),
    (r"^show\s+me\s+",                  ["Find ", "List ", "Return ", "Display "]),
    (r"^show\s+all\s+",                 ["List all ", "Find all ", "Return all ", "Display all "]),
    (r"^show\s+a\s+count\s+of\s+",      ["Count the ", "Find the number of ", "Return a count of ", "List the count of "]),
    (r"^show\s+the\s+",                 ["Find the ", "List the ", "Return the ", "Display the "]),
    (r"^show\s+",                       ["List ", "Find ", "Return ", "Display "]),
    (r"^list\s+all\s+",                 ["Show all ", "Find all ", "Return all ", "Display all "]),
    (r"^list\s+the\s+",                 ["Show the ", "Find the ", "Return the ", "Display the "]),
    (r"^list\s+",                       ["Show ", "Find ", "Return ", "Display "]),
    (r"^make\s+a\s+list\s+of\s+",       ["List ", "Show all ", "Find all ", "Return all "]),
    (r"^find\s+all\s+",                 ["List all ", "Show all ", "Return all ", "Retrieve all "]),
    (r"^find\s+the\s+",                 ["Show the ", "List the ", "Return the ", "Retrieve the "]),
    (r"^find\s+",                       ["Show ", "List ", "Return ", "Retrieve "]),
    (r"^get\s+the\s+",                  ["Find the ", "Show the ", "Return the ", "List the "]),
    (r"^get\s+",                        ["Find ", "Show ", "Return ", "List "]),
    (r"^retrieve\s+",                   ["Show ", "List ", "Find ", "Return "]),
    (r"^display\s+",                    ["Show ", "List ", "Find ", "Return "]),
    (r"^return\s+",                     ["Show ", "List ", "Find ", "Display "]),
    (r"^what\s+are\s+the\s+",           ["List the ", "Show the ", "Find the ", "Return the "]),
    (r"^what\s+are\s+all\s+the\s+",     ["List all ", "Show all ", "Find all ", "Return all "]),
    (r"^what\s+is\s+the\s+total\s+",    ["Find the total ", "Show the sum of ", "Return the total ", "Calculate the total "]),
    (r"^what\s+is\s+the\s+average\s+",  ["Find the average ", "Show the mean ", "Return the mean ", "Calculate the average "]),
    (r"^what\s+is\s+the\s+",            ["Find the ", "Show the ", "Return the ", "List the "]),
    (r"^what\s+is\s+",                  ["Find ", "Show ", "Return ", "List "]),
    (r"^how\s+many\s+",                 ["Count the number of ", "What is the count of ", "Find the total number of ", "Return the number of "]),
    (r"^which\s+",                      ["Find the ", "List the ", "Show the ", "Return the "]),
    (r"^count\s+the\s+number\s+of\s+",  ["How many ", "Find the number of ", "Return the count of ", "List the count of "]),
    (r"^count\s+",                      ["How many ", "Find the number of ", "Return the count of "]),
    (r"^give\s+me\s+",                  ["Show ", "Find ", "List ", "Return "]),
    (r"^provide\s+",                    ["Show ", "Find ", "List ", "Return "]),
    (r"^calculate\s+",                  ["Find ", "Show ", "Return ", "Compute "]),
    (r"^identify\s+",                   ["Find ", "List ", "Show ", "Return "]),
    (r"^select\s+",                     ["Show ", "List ", "Find ", "Return "]),
]

def apply_stem_swap(question: str, rng: random.Random) -> str | None:
    q_lower = question.lower().strip()
    for pattern, replacements in STEM_RULES:
        m = re.match(pattern, q_lower)
        if m:
            chosen = rng.choice(replacements)
            rest = question[m.end():]
            if not rest:
                continue
            new_q = chosen + rest[0].lower() + rest[1:]
            new_q = new_q[0].upper() + new_q[1:]
            if new_q.rstrip("?. ").lower() != question.rstrip("?. ").lower():
                return new_q
    return None


# ---------------------------------------------------------------------------
# Inline phrase swaps (replace phrases anywhere in the question)
# ---------------------------------------------------------------------------

INLINE_RULES = [
    # (pattern, list of replacements)  — replacements exclude the original match
    (r"\bfor each\b",      ["per", "grouped by", "by"]),
    (r"\bper\b",           ["for each", "grouped by", "by"]),
    (r"\bgrouped by\b",    ["for each", "per", "by"]),
    (r"\bordered by\b",    ["sorted by", "ranked by"]),
    (r"\bsorted by\b",     ["ordered by", "ranked by"]),
    (r"\bfrom most to least\b", ["in descending order", "sorted descending"]),
    (r"\bfrom least to most\b", ["in ascending order", "sorted ascending"]),
    (r"\btotal number of\b",  ["count of", "number of"]),
    (r"\btotal\b",         ["overall", "aggregate", "combined"]),
    (r"\baverage\b",       ["mean", "avg"]),
    (r"\bmean\b",          ["average", "avg"]),
    (r"\bhighest\b",       ["maximum", "greatest", "largest"]),
    (r"\blargest\b",       ["highest", "maximum", "greatest"]),
    (r"\bgreatest\b",      ["highest", "maximum", "largest"]),
    (r"\blowest\b",        ["minimum", "smallest", "least"]),
    (r"\bsmallest\b",      ["lowest", "minimum", "least"]),
    (r"\bmost\b",          ["highest number of", "the greatest number of"]),
    (r"\bdistinct\b",      ["unique", "different"]),
    (r"\bunique\b",        ["distinct", "different"]),
    (r"\ball\b",           ["every", "each"]),
    (r"\bevery\b",         ["all", "each"]),
    (r"\bin the database\b", ["in the records", "in the data"]),
    (r"\bcount\b",         ["number", "total"]),
    (r"\bnumber\b",        ["count", "total"]),
]

def apply_inline_swap(question: str, rng: random.Random) -> str | None:
    # Collect all matching positions
    candidates = []
    for pattern, replacements in INLINE_RULES:
        for m in re.finditer(pattern, question, re.IGNORECASE):
            candidates.append((m.start(), m.end(), m.group(), replacements))
    if not candidates:
        return None
    start, end, matched, replacements = rng.choice(candidates)
    chosen = rng.choice(replacements)
    # Preserve original capitalization at match start
    if matched[0].isupper():
        chosen = chosen[0].upper() + chosen[1:]
    new_q = question[:start] + chosen + question[end:]
    if new_q.lower() != question.lower():
        return new_q
    return None


# ---------------------------------------------------------------------------
# SQL-structure template questions
# ---------------------------------------------------------------------------

def _col_display(col: str) -> str:
    """Turn COLUMN_NAME into 'column name' for readability."""
    return col.replace("_", " ").lower()

def _table_display(table: str) -> str:
    """Turn TABLE_NAME into 'table name' for readability."""
    return table.replace("_", " ").lower()


def generate_sql_template(example: dict, rng: random.Random) -> str | None:
    sql     = example["gold_sql"]
    links   = example["schema_links"]
    p       = analyze_sql(sql)
    tables  = list(links.keys())
    all_cols = [c for cols in links.values() for c in cols]

    if not tables:
        return None

    t0 = _table_display(tables[0])
    c0 = _col_display(all_cols[0]) if all_cols else "value"

    # COUNT + GROUP BY → per-group count question
    if p["has_count"] and p["has_group_by"] and all_cols:
        group_col = _col_display(all_cols[-1])  # heuristic: last col is often the group
        options = [
            f"For each {group_col}, what is the count of {t0} records?",
            f"How many {t0} entries exist for each {group_col}?",
            f"Return the number of {t0} records grouped by {group_col}.",
            f"What is the total count of {t0} entries per {group_col}?",
        ]
        return rng.choice(options)

    # MAX
    if p["has_max"] and all_cols:
        options = [
            f"What is the maximum {c0} in the {t0} table?",
            f"Find the highest {c0} across all {t0} records.",
            f"Return the largest {c0} value from {t0}.",
        ]
        return rng.choice(options)

    # MIN
    if p["has_min"] and all_cols:
        options = [
            f"What is the minimum {c0} in the {t0} table?",
            f"Find the lowest {c0} across all {t0} records.",
            f"Return the smallest {c0} value from {t0}.",
        ]
        return rng.choice(options)

    # AVG
    if p["has_avg"] and all_cols:
        options = [
            f"What is the average {c0} in the {t0} table?",
            f"Calculate the mean {c0} across all {t0} records.",
            f"Find the mean value of {c0} from {t0}.",
        ]
        return rng.choice(options)

    # SUM
    if p["has_sum"] and all_cols:
        options = [
            f"What is the total {c0} across all {t0} records?",
            f"Find the sum of {c0} in {t0}.",
            f"Return the combined {c0} from all {t0} entries.",
        ]
        return rng.choice(options)

    # JOIN with at least 2 tables
    if p["has_join"] and len(tables) >= 2:
        t1 = _table_display(tables[1])
        join_cols = all_cols[:2] if len(all_cols) >= 2 else all_cols
        col_str = " and ".join(_col_display(c) for c in join_cols)
        options = [
            f"Find the {col_str} by joining {t0} with {t1}.",
            f"Return records from {t0} matched with {t1}.",
            f"What {col_str} can be found by combining {t0} and {t1}?",
        ]
        return rng.choice(options)

    # TOP / ORDER BY DESC → ranking question
    if p["has_top"] and p["is_desc"] and all_cols:
        options = [
            f"Return the top {t0} records by {c0} in descending order.",
            f"Which {t0} entries have the highest {c0}?",
            f"Find the {t0} records ranked by {c0} from most to least.",
        ]
        return rng.choice(options)

    # ORDER BY ASC
    if p["has_order_by"] and not p["is_desc"] and all_cols:
        options = [
            f"List all {t0} records sorted by {c0}.",
            f"Return {t0} entries ordered by {c0} ascending.",
            f"Find {t0} records ranked by {c0} from least to most.",
        ]
        return rng.choice(options)

    # COUNT alone
    if p["has_count"]:
        options = [
            f"How many records are in the {t0} table?",
            f"What is the total count of {t0} entries?",
            f"Count the number of rows in {t0}.",
        ]
        return rng.choice(options)

    # DISTINCT
    if p["has_distinct"] and all_cols:
        options = [
            f"What are the unique {c0} values in {t0}?",
            f"List all distinct {c0} entries from {t0}.",
            f"Find every different {c0} in the {t0} table.",
        ]
        return rng.choice(options)

    return None


# ---------------------------------------------------------------------------
# Prefix injection fallback
# ---------------------------------------------------------------------------

PREFIXES = [
    "In the database, ",
    "From the records, ",
    "Based on the data, ",
    "Using the available data, ",
    "According to the database, ",
    "From the stored data, ",
]

def apply_prefix(question: str, rng: random.Random) -> str:
    prefix = rng.choice(PREFIXES)
    q = question[0].lower() + question[1:]
    return prefix + q


# ---------------------------------------------------------------------------
# Main variant generator
# ---------------------------------------------------------------------------

def generate_variant(example: dict, attempt: int = 0) -> str | None:
    """
    Try all strategies in order of quality. Return the first variant that
    differs meaningfully from the original question, or None on total failure.
    attempt controls the random seed so repeated calls yield different results.
    """
    rng = random.Random(example["question_id"] * 31 + attempt * 97)
    original = example["question"].strip()

    strategies = [
        lambda: apply_stem_swap(original, rng),
        lambda: apply_inline_swap(original, rng),
        lambda: generate_sql_template(example, rng),
        lambda: apply_prefix(original, rng),
    ]

    for fn in strategies:
        variant = fn()
        if variant and variant.strip().lower() != original.lower():
            return variant.strip()

    return None


# ---------------------------------------------------------------------------
# Sampling with smart weights
# ---------------------------------------------------------------------------

def compute_weights(data: list[dict]) -> list[float]:
    db_counts = Counter(d["db_id"] for d in data)
    weights = []
    for d in data:
        n_tables = len(d["schema_links"])
        db_freq  = db_counts[d["db_id"]]
        # reward multi-table (JOIN) examples and under-represented DBs
        w = (1.0 + 2.0 * (n_tables >= 2)) * (1.0 / db_freq)
        weights.append(w)
    return weights


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    with open(TRAIN_JSON) as f:
        data = json.load(f)

    weights = compute_weights(data)
    total_w = sum(weights)
    norm_w  = [w / total_w for w in weights]

    rng_sample = random.Random(42)
    max_id = max(d["question_id"] for d in data)

    augmented = []
    seen_questions: set[str] = {d["question"].strip().lower() for d in data}
    attempt_tracker: dict[int, int] = {}  # question_id -> next attempt number

    # Try up to 5x target to fill TARGET_N
    max_draws = TARGET_N * 5
    draws = 0

    while len(augmented) < TARGET_N and draws < max_draws:
        draws += 1
        idx = rng_sample.choices(range(len(data)), weights=norm_w, k=1)[0]
        example = data[idx]

        attempt = attempt_tracker.get(example["question_id"], 0)
        attempt_tracker[example["question_id"]] = attempt + 1

        variant = generate_variant(example, attempt=attempt)
        if variant is None:
            continue

        key = variant.strip().lower()
        if key in seen_questions:
            continue

        seen_questions.add(key)
        new_id = max_id + len(augmented) + 1

        augmented.append({
            "question_id":  new_id,
            "db_id":        example["db_id"],
            "question":     variant,
            "gold_sql":     example["gold_sql"],
            "schema_links": example["schema_links"],
        })

    print(f"Generated {len(augmented)} augmented examples (target={TARGET_N}, draws={draws})")

    # Breakdown by strategy type
    db_breakdown = Counter(d["db_id"] for d in augmented)
    print("\nDB distribution in augmented set:")
    for db, cnt in sorted(db_breakdown.items(), key=lambda x: -x[1]):
        print(f"  {db}: {cnt}")

    multi_table = sum(1 for d in augmented if len(d["schema_links"]) >= 2)
    print(f"\nMulti-table examples: {multi_table}/{len(augmented)}")

    with open(OUT_JSON, "w") as f:
        json.dump(augmented, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {OUT_JSON}")


if __name__ == "__main__":
    main()
