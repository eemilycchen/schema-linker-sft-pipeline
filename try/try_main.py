import argparse
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SYSTEM_PROMPT = '''You are a schema-linking model.\n
Given a database schema and a natural-language question, output only valid JSON.
The JSON must map table names to lists of referenced column names.
Include tables even when no specific columns are referenced, using an empty list.
Do not include explanations, markdown, or extra text.

Example output:
{"TableName1": ["Col1", "Col2"], "TableName2": ["Col3"]}
'''

# load the checkpoint
base_model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
checkpoint_path = '/home/chc016/rapidfireai/rapidfire_experiments/schema-linking-qwen-baseline_12/runs/2/checkpoints/final_checkpoint'
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    torch_dtype="auto",
)
model = PeftModel.from_pretrained(base_model, checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model.eval()

def load_schema(db_id, schemas_dir='../schemas'):
    # change to the correct file name
    fname = db_id.replace(' ', '_').replace('/', '_') + '.json'
    # load the json file
    with open(f'{schemas_dir}/{fname}') as f:
        s = json.load(f)
    # to store table name and column mapping
    schema = {t: [] for t in s['table_names_original']}
    for tidx, cname in s['column_names_original']:
        # skip the synthetic '*' entry
        if tidx == -1:
            continue
        schema[s['table_names_original'][tidx]].append(cname)
    # {table_name: [col, col, ...]}
    return schema

def build_prompt(question, schema_text):
    return (
        f'Question:\n{question}\n'
        f'Schema:\n{schema_text}\n'
        'Return only schema_links JSON.'
    )

def predict_schema_links(question, db_id, schemas_dir):
    """Replace this with your model's inference logic."""
    # load the schemas
    schema = load_schema(db_id, schemas_dir)
    # build user prompt
    schema_text = '\n'.join(
        f'- {table}: {", ".join(columns)}'
        for table, columns in schema.items()
    )
    user_prompt = build_prompt(question, schema_text)    
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]
    # create the prompt
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    # tokenize to be able to input to the model
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    # generate the answer
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False, # deterministically sample
        pad_token_id=tokenizer.eos_token_id,
    )
    # decode the tokenized output
    generated_ids = outputs[0][inputs['input_ids'].shape[-1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return extract_schema_links(generated_text)

def extract_schema_links(text):
    import json, re
    text = str(text).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # Prefer the first JSON-looking object in the generated output.
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {}
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    if not isinstance(obj, dict):
        return {}
    cleaned = {}
    for table, cols in obj.items():
        if isinstance(cols, list):
            cleaned[str(table)] = [str(c) for c in cols]
        else:
            cleaned[str(table)] = []
    return cleaned

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',  required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--schemas_dir', default='../schemas')
    args = ap.parse_args()

    with open(args.input) as f:
        items = json.load(f)
    preds = []
    for idx, it in enumerate(items):
        print(f'\rgenerating answer [{idx + 1} / {len(items)}]', end='')
        links = predict_schema_links(it['question'], it['db_id'], args.schemas_dir)
        preds.append({'question_id': it['question_id'], 'schema_links': links})
    with open(args.output, 'w') as f:
        json.dump(preds, f, indent=2)
    print(f"Wrote {len(preds)} predictions to {args.output}")


if __name__ == '__main__':
    main()