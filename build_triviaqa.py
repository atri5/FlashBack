# build_triviaqa_for_flashback.py
from datasets import load_dataset, DatasetDict

out_dir = r"C:\Users\atrip\Davis\Classes\ECS-189G-2\ECS-189G-NLP-Final-Project\reference_code\datasets\triviaqa"

ds = load_dataset("mandarjoshi/trivia_qa", "rc")  # official HF TriviaQA RC config

def to_text(ex):
    q = ex.get("question", "")
    ctx_parts = []

    for k in ("entity_pages", "search_results"):
        block = ex.get(k, {})
        if isinstance(block, dict):
            # common fields in TriviaQA HF versions
            for field in ("wiki_context", "search_context", "title"):
                vals = block.get(field, [])
                if isinstance(vals, list):
                    ctx_parts.extend([v for v in vals if isinstance(v, str)])

    text = (q + "\n" + "\n".join(ctx_parts)).strip()
    return {"text": text}

train = ds["train"].map(to_text, remove_columns=ds["train"].column_names)
val = ds["validation"].map(to_text, remove_columns=ds["validation"].column_names)

# repo's data2jsonl.py expects train/validation/test, so create test from validation
tmp = val.train_test_split(test_size=0.5, seed=42)

out = DatasetDict({
    "train": train,
    "validation": tmp["train"],
    "test": tmp["test"],
})
out.save_to_disk(out_dir)
print("saved:", out_dir)
