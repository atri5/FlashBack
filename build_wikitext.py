from datasets import load_dataset, DatasetDict

# Toggle this to quickly switch between a small build and full build.
USE_DOWNSAMPLE = True
SEED = 42

# Full output path (all rows).
FULL_OUT_DIR = r"C:\Users\atrip\Davis\Classes\ECS-189G-2\ECS-189G-NLP-Final-Project\reference_code\datasets\wikitext103"
# Smaller output path for faster iteration / repeated eval sampling.
SMALL_OUT_DIR = r"C:\Users\atrip\Davis\Classes\ECS-189G-2\ECS-189G-NLP-Final-Project\reference_code\datasets\wikitext103_small"

# Per-split limits used only when USE_DOWNSAMPLE=True.
SPLIT_LIMITS = {
    "train": 50000,
    "validation": 20000,
    "test": 5000,
}


def maybe_downsample(split_ds, split_name: str):
    """Return full split or a shuffled subset depending on USE_DOWNSAMPLE."""
    if not USE_DOWNSAMPLE:
        return split_ds
    limit = SPLIT_LIMITS.get(split_name)
    if limit is None:
        return split_ds
    keep = min(limit, len(split_ds))
    return split_ds.shuffle(seed=SEED).select(range(keep))


def main():
    ds = load_dataset("wikitext", "wikitext-103-v1")
    out = DatasetDict(
        {
            "train": maybe_downsample(ds["train"], "train"),
            "validation": maybe_downsample(ds["validation"], "validation"),
            "test": maybe_downsample(ds["test"], "test"),
        }
    )
    out_dir = SMALL_OUT_DIR if USE_DOWNSAMPLE else FULL_OUT_DIR
    out.save_to_disk(out_dir)
    print("saved:", out_dir)
    print({k: len(v) for k, v in out.items()})


if __name__ == "__main__":
    main()
