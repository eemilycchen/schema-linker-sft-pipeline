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


# found in project 2 statement
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
    """Macro table/column P/R/F1 + leaderboard_score.

    Both lists contain JSON strings (or plain dicts for references).
    """
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


# ── TRL trainer hooks ─────────────────────────────────────────────────────────

def preprocess_logits_for_metrics(logits, labels):
    """
    TRL hook called *before* compute_metrics on every eval batch.

    Converts the 3-D logit tensor (batch x seq_len x vocab_size) to argmax
    token-ID predictions (batch x seq_len) on the GPU, before transfer to CPU.
    This avoids materialising a (batch x seq_len x 32000) float32 tensor in
    CPU RAM, which would OOM on a 24 GB MIG slice.

    Wire it up alongside make_compute_metrics:

        RFSFTConfig(
            ...
            compute_metrics=make_compute_metrics(tokenizer),
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
    """
    if isinstance(logits, tuple):
        logits = logits[0]           # some models return (logits, past_kv, ...)
    return logits.argmax(dim=-1)     # (batch, seq_len) int64, still on GPU


def make_compute_metrics(tokenizer):
    """
    Factory: returns a compute_metrics callable for TRL SFTTrainer.

    Pass the tokenizer once at notebook setup time — the returned function
    is a closure that carries it without any global state.

    WHY A FACTORY?
    ──────────────
    HuggingFace's compute_metrics signature is fixed to (eval_preds) -> dict.
    There is no way to pass extra arguments, so a closure is the cleanest
    way to inject the tokenizer.

    THE OFF-BY-ONE (causal LM alignment)
    ─────────────────────────────────────────────────────────────────────
    A causal LM produces logits[j] = the distribution over vocabulary for
    predicting the *next* token (position j+1), given the prefix 0..j.

    After preprocess_logits_for_metrics converts logits -> argmax token IDs:

        predictions[j]   predicts the token at position j+1
        labels[j+1]      is the gold token at position j+1  (or -100)

    Alignment rule:   predictions[:-1]  <->  labels[1:]

    The completion tokens are positions where labels[1:] != -100.
    TRL's DataCollatorForCompletionOnlyLM sets all prompt positions to -100.

    TEACHER-FORCING NOTE
    ────────────────────
    During validation the model sees the correct prefix at every step
    (teacher forcing), so these scores will be somewhat higher than what
    main.py produces via free-form autoregressive generation.  They are
    still a reliable relative indicator: if leaderboard_score improves
    epoch-over-epoch here, the model is genuinely learning.
    ─────────────────────────────────────────────────────────────────────
    """

    def compute_metrics(eval_preds) -> dict:
        import numpy as np

        predictions, labels = eval_preds

        # Handle raw logits (3-D) if preprocess_logits_for_metrics was not wired
        # predictions shape: (batch, seq_len, vocab_size) float -> argmax to int
        if hasattr(predictions, "ndim") and predictions.ndim == 3:
            predictions = predictions.argmax(axis=-1)

        decoded_preds: list[str] = []
        decoded_labels: list[str] = []

        for pred_row, label_row in zip(predictions, labels):
            # Apply the causal-LM off-by-one alignment:
            #   pred_row[j]    predicts the token at position j+1
            #   label_row[j+1] is the gold token at position j+1
            #
            # Trim: drop the last prediction (nothing to predict after EOS)
            #       drop the first label  (it's always a prompt token / -100)
            pred_aligned  = pred_row[:-1]   # (seq_len - 1,)
            label_aligned = label_row[1:]   # (seq_len - 1,)

            # Completion-token mask: True where TRL preserved a real label
            mask = label_aligned != -100

            if mask.any():
                pred_tokens  = pred_aligned[mask].tolist()
                label_tokens = label_aligned[mask].tolist()

                decoded_preds.append(
                    tokenizer.decode(pred_tokens,  skip_special_tokens=True)
                )
                decoded_labels.append(
                    tokenizer.decode(label_tokens, skip_special_tokens=True)
                )
            else:
                # No completion tokens found (sequence was entirely prompt
                # tokens after truncation). Emit empty JSON so the scorer
                # returns 0 rather than crashing.
                decoded_preds.append("{}")
                decoded_labels.append("{}")

        return schema_link_metrics(decoded_preds, decoded_labels)

    return compute_metrics


# ── REMOVED — the broken original ────────────────────────────────────────────
# def compute_metrics_for_rf(eval_preds) -> dict:
#     predictions, labels = eval_preds
#     return schema_link_metrics(list(predictions), list(labels))
#                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   BUG: predictions is a numpy int array of token IDs, NOT a list of JSON
#   strings.  str(np.array([23, 456, ...])) = "[ 23 456 ...]" which contains
#   no { } block, so parse_json_output always returns {} for every example.
#   The only non-zero contribution comes from 3 wildcard queries in
#   validation.json (empty gold columns = empty pred columns -> col F1 = 1),
#   yielding exactly leaderboard_score = 0.0149.