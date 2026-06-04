#!/usr/bin/env python3
"""
main.py — Schema-linking inference script for CSE/DSC 234 Project 2.

Usage:
    python3 main.py --input <input_json> --output <output_json>
                    [--schemas_dir ./schemas] [--batch_size 8]
                    [--checkpoint_path ./adapter]
                    [--model_name Qwen/Qwen2.5-1.5B-Instruct]
"""
import argparse, json, re, sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent / "try"))
from schema_formatting import serialize_schema_minimal  # make sure this matches training!


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",           required=True)
    p.add_argument("--output",          required=True)
    p.add_argument("--schemas_dir",     default="./schemas")
    p.add_argument("--batch_size",      type=int, default=8)
    p.add_argument("--checkpoint_path", default="./adapter")
    p.add_argument("--model_name",      default="Qwen/Qwen2.5-1.5B-Instruct")
    return p.parse_args()


# ---- Schema helpers ---------------------------------------------------------
_schema_cache = {}

def load_schema(db_id, schemas_dir="./schemas"):
    if db_id not in _schema_cache:
        fname = db_id.replace(" ", "_") + ".json"
        with open(Path(schemas_dir) / fname) as f:
            _schema_cache[db_id] = json.load(f)
    return _schema_cache[db_id]



def _parse_json_output(text):
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


def _tokenize_identifier(name):
    """Split a column/table name into lowercase tokens.
    'OrderDate' -> ['order', 'date'], 'CASEID' -> ['caseid'], 'crash_id' -> ['crash', 'id']
    """
    # Split camelCase: insert space before each uppercase letter that follows a lowercase
    name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    # Split on underscores/spaces and lowercase everything
    return [t for t in re.split(r'[_\s]+', name.lower()) if t]


def _bm25_column_fallback(question, t_lower, col_lower2orig, top_k=3):
    """Return top-k columns by token overlap with the question.
    Returns [] if no column shares any token with the question.
    """
    q_tokens = Counter(re.findall(r'[a-z]+', question.lower()))
    if not q_tokens:
        return []

    scores = []
    for col_lower, col_orig in col_lower2orig.get(t_lower, {}).items():
        col_tokens = _tokenize_identifier(col_lower)
        score = sum(q_tokens.get(tok, 0) for tok in col_tokens)
        scores.append((score, col_orig))

    scores.sort(key=lambda x: -x[0])
    return [col for score, col in scores[:top_k] if score > 0]


def postprocess_output(raw_text, db_id, schemas_dir="./schemas", question=""):
    schema = load_schema(db_id, schemas_dir)
    table_lower2orig = {t.lower(): t for t in schema["table_names_original"]}

    col_lower2orig = {}
    for (t_idx, cname) in schema["column_names_original"]:
        if t_idx == -1:  # skip synthetic [-1, "*"] wildcard entry
            continue
        tname = schema["table_names_original"][t_idx]
        col_lower2orig.setdefault(tname.lower(), {})[cname.lower()] = cname

    pred = _parse_json_output(raw_text)
    if not pred:
        return {}  # parse failure: return empty rather than poisoning precision

    validated = {}
    for pred_table, pred_cols in pred.items():
        t_lower = pred_table.lower()
        if t_lower not in table_lower2orig:
            continue  # hallucinated table — drop it
        orig_table = table_lower2orig[t_lower]
        valid_cols = []
        flat_cols = []
        if not isinstance(pred_cols, list):
            pred_cols = [] # guard against anything that's not a list
        for c in pred_cols:
            if isinstance(c, list):  # flatten nested lists e.g. [["col1", "col2"]] -> ["col1", "col2"]
                flat_cols.extend(c)
            elif isinstance(c, str):
                flat_cols.append(c)
        for c in flat_cols:
            if not isinstance(c, str):
                continue
            col_map = col_lower2orig.get(t_lower, {})
            if c.lower() in col_map:
                valid_cols.append(col_map[c.lower()])

        if not valid_cols and question:
            valid_cols = _bm25_column_fallback(question, t_lower, col_lower2orig)

        validated[orig_table] = valid_cols
    return validated

# before it wsa 0.2533 changing this for format errors and putting for answering wrong tables
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


def build_prompt(question, db_id, schemas_dir="./schemas", tokenizer=None, max_length=2048):
    schema = load_schema(db_id, schemas_dir)
    lines = serialize_schema_minimal(schema).split("\n")

    if tokenizer is not None:
        # Reserve tokens for system prompt + question + formatting overhead
        overhead_msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Database schema:\n\n\nQuestion: {question}"},
        ]
        overhead = len(tokenizer.encode(
            tokenizer.apply_chat_template(overhead_msgs, tokenize=False, add_generation_prompt=True)
        ))
        budget = max_length - overhead - 64  # 64 buffer for generation

        # Add table lines until budget is exhausted
        kept, used = [], 0
        for line in lines:
            line_toks = len(tokenizer.encode(line))
            if used + line_toks > budget:
                break
            kept.append(line)
            used += line_toks
        schema_s = "\n".join(kept) if kept else "\n".join(lines[:5])
    else:
        schema_s = "\n".join(lines)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Database schema:\n{schema_s}\n\nQuestion: {question}"},
    ]


# ---- Main -------------------------------------------------------------------
def main():
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Loading base model: {args.model_name}")
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name, device_map="auto", dtype="auto", use_cache=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left" # keep the questions

    print(f"Loading LoRA adapter from: {args.checkpoint_path}")
    model = PeftModel.from_pretrained(base, args.checkpoint_path)
    model.eval()

    with open(args.input) as f:
        inputs = json.load(f)

    outputs = []
    for i in range(0, len(inputs), args.batch_size):
        batch = inputs[i : i + args.batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                build_prompt(ex["question"], ex["db_id"], args.schemas_dir, tokenizer=tokenizer),
                tokenize=False, add_generation_prompt=True
            )
            for ex in batch
        ]
        enc = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=2048 # consistent with training
        ).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=512, # consistent with training
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for ex, input_ids_row, gen_ids_row in zip(batch, enc["input_ids"], gen):
            new_tokens = gen_ids_row[len(input_ids_row):]
            raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            # debug
            print(f"  RAW[{ex['question_id']}]: {raw_text[:200]}")  # debug
            schema_links = postprocess_output(raw_text, ex["db_id"], args.schemas_dir, question=ex["question"])
            outputs.append({"question_id": ex["question_id"], "schema_links": schema_links})
        print(f"  Processed {min(i + args.batch_size, len(inputs))}/{len(inputs)} questions")

    with open(args.output, "w") as f:
        json.dump(outputs, f, indent=2)
    print(f"Saved {len(outputs)} predictions to {args.output}")


if __name__ == "__main__":
    main()
