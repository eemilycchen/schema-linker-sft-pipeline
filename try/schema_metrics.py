"""Schema-linking eval metrics for RapidFire compute_metrics callback."""

from __future__ import annotations

import json
import re


def parse_json_output(text: str) -> dict:
    """Strip markdown fences and parse first JSON object."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {}

# from project 2 statement
def prf1(pred_set: set, gold_set: set) -> tuple[float, float, float]:
    if not gold_set and not pred_set:
        return 1.0, 1.0, 1.0
    if not gold_set:
        return 0.0, 1.0, 0.0
    if not pred_set:
        return 0.0, 0.0, 0.0
    tp = len(pred_set & gold_set)
    p = tp / len(pred_set)
    r = tp / len(gold_set)
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def schema_link_metrics(predictions: list[str], references: list[str]) -> dict:
    """Macro table/column P/R/F1 + leaderboard_score."""
    t_p_list, t_r_list, t_f1_list = [], [], []
    c_p_list, c_r_list, c_f1_list = [], [], []

    for pred_str, gold_str in zip(predictions, references):
        pred_links = parse_json_output(pred_str)
        gold_links = json.loads(gold_str) if isinstance(gold_str, str) else gold_str

        pred_tables = {t.lower() for t in pred_links}
        gold_tables = {t.lower() for t in gold_links}
        tp, tr, tf = prf1(pred_tables, gold_tables)
        t_p_list.append(tp)
        t_r_list.append(tr)
        t_f1_list.append(tf)

        pred_cols = {
            f"{t.lower()}.{c.lower()}"
            for t, cols in pred_links.items()
            for c in (cols or [])
        }
        gold_cols = {
            f"{t.lower()}.{c.lower()}"
            for t, cols in gold_links.items()
            for c in (cols or [])
        }
        cp, cr, cf = prf1(pred_cols, gold_cols)
        c_p_list.append(cp)
        c_r_list.append(cr)
        c_f1_list.append(cf)

    def _mean(lst: list[float]) -> float:
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    pt, rt, f1t = _mean(t_p_list), _mean(t_r_list), _mean(t_f1_list)
    pc, rc, f1c = _mean(c_p_list), _mean(c_r_list), _mean(c_f1_list)
    table_score = (pt + rt + f1t) / 3
    col_score = (pc + rc + f1c) / 3
    leaderboard = round(0.5 * table_score + 0.5 * col_score, 4)

    return {
        "table_P": pt,
        "table_R": rt,
        "table_F1": f1t,
        "col_P": pc,
        "col_R": rc,
        "col_F1": f1c,
        "leaderboard_score": leaderboard,
    }


def compute_metrics_for_rf(eval_preds) -> dict:
    """Adapter for RFSFTConfig.compute_metrics."""
    predictions, labels = eval_preds
    return schema_link_metrics(list(predictions), list(labels))
