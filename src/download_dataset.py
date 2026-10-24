"""Download and inspect the Bitext customer support dataset."""
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
FILENAME = "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=FILENAME,
                          local_dir=raw_dir)
    frame = pd.read_csv(path)
    print("Shape:", frame.shape)
    print("Columns:", frame.columns.tolist())
    print(frame.head(3).to_string(index=False))
    destination = ROOT / "data" / "customer_support_raw.csv"
    frame.to_csv(destination, index=False)
    print("Saved", destination)


if __name__ == "__main__":
    main()
