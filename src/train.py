"""LoRA supervised fine-tuning for Qwen2.5-0.5B-Instruct."""
import argparse
try:
    import torch
except ImportError as exc:
    raise RuntimeError("PyTorch is not installed. Install a CUDA-enabled Windows build using the official PyTorch selector, then reinstall requirements.txt.") from exc
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from inference import _model_source

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "customer-support-lora"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, help="Cap training steps for a smoke run; omit for the full three-epoch run.")
    cli = parser.parse_args()
    if cli.max_steps is not None and cli.max_steps < 1:
        parser.error("--max-steps must be a positive integer")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected. Check the NVIDIA driver and install a CUDA-enabled PyTorch build.")
    train_path, validation_path = ROOT / "data/processed/train.jsonl", ROOT / "data/processed/validation.jsonl"
    if not train_path.exists() or not validation_path.exists():
        raise FileNotFoundError("Processed splits are missing. Run src/prepare_data.py first.")
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
    model_source = _model_source()
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_source, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.config.use_cache = False
    dataset = load_dataset("json", data_files={"train": str(train_path), "validation": str(validation_path)},
                           cache_dir=str(ROOT / ".cache" / "datasets"))
    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    smoke_run = cli.max_steps is not None
    args = SFTConfig(output_dir=str(OUTPUT_DIR), num_train_epochs=3, max_steps=cli.max_steps or -1,
                     per_device_train_batch_size=1,
                     per_device_eval_batch_size=1, gradient_accumulation_steps=8, learning_rate=2e-4,
                     fp16=True, gradient_checkpointing=True, optim="adamw_torch", max_length=256,
                     logging_steps=10, eval_strategy="no" if smoke_run else "epoch",
                     save_strategy="steps" if smoke_run else "epoch",
                     save_steps=cli.max_steps or 500, save_total_limit=2,
                     report_to="none", seed=42, dataset_kwargs={"skip_prepare_dataset": False})
    trainer = SFTTrainer(model=model, args=args, train_dataset=dataset["train"],
                         eval_dataset=dataset["validation"], processing_class=tokenizer, peft_config=lora)
    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("Adapter saved:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
