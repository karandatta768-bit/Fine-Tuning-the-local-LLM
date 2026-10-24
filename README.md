# Local Customer Support LLM

An experimental local customer-support suggestion system using Qwen2.5-0.5B-Instruct and LoRA supervised fine-tuning. It classifies messages and drafts a dataset-grounded response as JSON. It does not access orders, authorize refunds, cancel orders, or perform other business actions. Treat every generated response as a suggestion for a person to review.

## Workflow

1. Create and activate a Python 3.11 virtual environment. Install a CUDA-enabled PyTorch build using the official PyTorch selector for Windows before installing the remaining packages; a generic CPU wheel will not enable the RTX GPU.
2. Install the remaining packages with `python -m pip install -r requirements.txt`.
3. Review the current Hugging Face dataset card and license. Download and inspect the CSV with `python src/download_dataset.py`.
4. Create deterministic stratified splits with `python src/prepare_data.py`.
5. Run `python src/evaluate.py` before training to record a base-model baseline on the held-out test split.
6. Verify one bounded training run with `python src/train.py --max-steps 100`, then run the documented full three epochs with `python src/train.py`. Both require CUDA and enough GPU memory; even the small Qwen model is slow on an RTX 3050 Ti laptop.
7. Rerun `python src/evaluate.py` to compare base and adapter results in `results/`.
8. Launch the local UI with `streamlit run app.py`.

The downloaded dataset is synthetic/curated. Keep its license and provenance with any distribution. The test split should remain held out during training and tuning. A deterministic per-intent splitter keeps every intent in all three splits without requiring scikit-learn. Metrics are generated from the actual local run and are not prefilled claims.

## Layout

- `src/download_dataset.py`: download and inspect source CSV
- `src/prepare_data.py`: clean and split source data
- `src/train.py`: LoRA SFT
- `src/inference.py`: local base/adapter inference with JSON validation and safe fallback
- `src/evaluate.py`: baseline and adapter test evaluation
- `app.py`: Streamlit suggestion interface

Large model weights and adapters are kept out of source control. Never put private customer records, credentials, or API keys in the repository.
