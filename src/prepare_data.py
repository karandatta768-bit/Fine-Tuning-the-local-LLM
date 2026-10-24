"""Clean the source CSV and make deterministic stratified JSONL splits."""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "customer_support_raw.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
REQUIRED = ["instruction", "category", "intent", "response"]


def convert_row(row) -> dict:
    target = {"category": row["category"], "intent": row["intent"], "response": row["response"]}
    return {"messages": [
        {"role": "user", "content": row["instruction"]},
        {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
    ]}


def main() -> None:
    import pandas as pd

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}. Run src/download_dataset.py first.")
    df = pd.read_csv(INPUT_FILE)
    missing = [column for column in REQUIRED if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset columns missing: {missing}. Actual columns: {df.columns.tolist()}")
    df = df.dropna(subset=REQUIRED).drop_duplicates(subset=["instruction", "response"]).copy()
    for column in REQUIRED:
        df[column] = df[column].astype(str).str.strip()
    df = df[(df.instruction != "") & (df.response != "") & (df.intent != "") & (df.category != "")]
    df = df.drop_duplicates(subset=["instruction"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No usable rows remain after cleaning.")
    counts = df.intent.value_counts()
    if counts.min() < 3:
        raise ValueError("Each intent needs at least 3 examples for stratified train/validation/test splits. "
                         f"Smallest intent counts: {counts[counts < 3].to_dict()}")
    print("Rows after cleaning:", len(df))
    print("Unique intents:", df.intent.nunique())
    print(df.intent.value_counts().to_string())
    buckets = {"train": [], "validation": [], "test": []}
    rng = random.Random(42)
    for _, group in df.groupby("intent", sort=True):
        indices = list(group.index)
        rng.shuffle(indices)
        n = len(indices)
        n_train = max(1, round(n * 0.8))
        n_validation = max(1, round((n - n_train) / 2))
        if n_train + n_validation >= n:
            n_train, n_validation = n - 2, 1
        buckets["train"].extend(indices[:n_train])
        buckets["validation"].extend(indices[n_train:n_train + n_validation])
        buckets["test"].extend(indices[n_train + n_validation:])
    train, validation, test = (df.loc[buckets[key]].sample(frac=1, random_state=42)
                               for key in ("train", "validation", "test"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("validation", validation), ("test", test)):
        path = OUTPUT_DIR / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for _, row in split.iterrows():
                handle.write(json.dumps(convert_row(row), ensure_ascii=False) + "\n")
        print(name, len(split), "->", path)


if __name__ == "__main__":
    main()
