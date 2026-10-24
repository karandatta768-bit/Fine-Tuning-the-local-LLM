"""Safe local inference for the base model or a trained LoRA adapter."""
import json
from pathlib import Path
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
try:
    import torch
except ImportError as exc:
    raise RuntimeError("PyTorch is not installed. Install a PyTorch build using the official selector before running inference.") from exc
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.response_schema import is_valid_response, normalize_response, resolve_placeholders

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = ROOT / "outputs" / "customer-support-lora"
SYSTEM_PROMPT = (
    "You are a customer support assistant. Return exactly one JSON object with string keys "
    "category, intent, response. Be polite and concise. Ask for missing information. Never claim "
    "a refund, cancellation, or order change has happened unless confirmed by a trusted system. "
    "Never request passwords, OTPs, or full payment-card details. Escalate sensitive or uncertain cases."
)
_MODELS = {}
_TOKENIZER = None
_ALLOWED_LABELS = None


def _model_source() -> str:
    """Prefer a complete cached snapshot to avoid unnecessary Hub API calls."""
    try:
        return snapshot_download(repo_id=BASE_MODEL, local_files_only=True)
    except (LocalEntryNotFoundError, FileNotFoundError):
        return BASE_MODEL


def _allowed_labels() -> tuple[set[str], set[str]]:
    global _ALLOWED_LABELS
    if _ALLOWED_LABELS is None:
        path = ROOT / "data" / "processed" / "train.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Training split not found: {path}. Run src/prepare_data.py first.")
        categories, intents = set(), set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                target = json.loads(json.loads(line)["messages"][1]["content"])
                categories.add(target["category"])
                intents.add(target["intent"])
        _ALLOWED_LABELS = (categories, intents)
    return _ALLOWED_LABELS


def load_model(use_adapter: bool = True):
    global _TOKENIZER
    if use_adapter in _MODELS:
        return _MODELS[use_adapter], _TOKENIZER
    adapter = use_adapter and (ADAPTER_PATH / "adapter_config.json").exists()
    if use_adapter and not adapter:
        raise FileNotFoundError(f"LoRA adapter not found at {ADAPTER_PATH}. Train first or use the base model.")
    model_source = _model_source()
    _TOKENIZER = AutoTokenizer.from_pretrained(model_source)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    if adapter:
        model = AutoModelForCausalLM.from_pretrained(model_source, torch_dtype=dtype, device_map="auto")
        model = PeftModel.from_pretrained(model, str(ADAPTER_PATH))
    else:
        model = AutoModelForCausalLM.from_pretrained(model_source, torch_dtype=dtype,
                  device_map="auto" if torch.cuda.is_available() else None)
        if not torch.cuda.is_available():
            model.to("cpu")
    model.eval()
    _MODELS[use_adapter] = model
    return model, _TOKENIZER


def answer_customer(message: str, use_adapter: bool = True) -> dict:
    if not isinstance(message, str) or not message.strip():
        return {"error": "empty_message", "needs_human_review": True}
    model, tokenizer = load_model(use_adapter)
    categories, intents = _allowed_labels()
    label_instruction = (
        f" Use exactly one category from {', '.join(sorted(categories))} and exactly one intent "
        f"from {', '.join(sorted(intents))}. Do not invent labels. Never output template tokens such as "
        "{{Order Number}}; use details the customer already provided and do not invent missing details."
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT + label_instruction}, {"role": "user", "content": message.strip()}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=160, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
    raw = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    try:
        result = json.loads(raw)
        result = normalize_response(result, categories, intents)
        if not is_valid_response(result, categories, intents):
            raise ValueError("Output has an unknown label or does not match the required schema")
        return resolve_placeholders(result, message)
    except (json.JSONDecodeError, ValueError):
        return {"error": "invalid_model_output", "needs_human_review": True,
                "response": "I'm unable to prepare a reliable response. Please ask a support agent to review this request."}


if __name__ == "__main__":
    while True:
        message = input("\nCustomer (or 'exit'): ").strip()
        if message.lower() in {"exit", "quit"}:
            break
        print(json.dumps(answer_customer(message), indent=2, ensure_ascii=False))
