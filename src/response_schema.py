"""Validation rules for dataset-derived assistant responses."""
import re

FIELDS = {"category", "intent", "response"}
INTENT_ALIASES = {"delivery_status": "track_order"}


def normalize_response(result: object, categories: set[str], intents: set[str]) -> object:
    if isinstance(result, dict) and result.get("intent") in INTENT_ALIASES:
        result = {**result, "intent": INTENT_ALIASES[result["intent"]]}
    return result


def resolve_placeholders(result: dict, message: str) -> dict:
    """Fill known order-number templates; replace any remaining templates safely."""
    response = result["response"]
    order_number = re.search(
        r"\border\s*(?:number|no\.?|#|id)\s*(?:is\s*)?#?\s*([A-Z0-9][A-Z0-9-]{2,})\b|\border\s+is\s+#?([A-Z0-9][A-Z0-9-]{2,})\b",
        message,
        re.IGNORECASE,
    )
    if order_number:
        number = next(group for group in order_number.groups() if group)
        response = re.sub(r"\{\{\s*Order\s+Number\s*\}\}", number, response, flags=re.IGNORECASE)
    if re.search(r"\{\{.*?\}\}", response):
        response = "I can’t provide a complete suggested reply because some required details are missing. Please have a support agent review this request."
    if result["intent"] == "track_order":
        if order_number:
            number = next(group for group in order_number.groups() if group)
            response = (
                f"Thanks for sharing order number {number}. I cannot access live tracking details here. "
                "Please check the tracking link in your shipping confirmation email or contact support for an update."
            )
        else:
            response = (
                "I cannot access live tracking details here. Please check the tracking link in your shipping "
                "confirmation email or contact support with your order number for an update."
            )
    return {**result, "response": response}


def is_valid_response(result: object, categories: set[str], intents: set[str]) -> bool:
    return (
        isinstance(result, dict)
        and set(result) == FIELDS
        and all(isinstance(result[field], str) and result[field].strip() for field in FIELDS)
        and result["category"] in categories
        and result["intent"] in intents
    )
