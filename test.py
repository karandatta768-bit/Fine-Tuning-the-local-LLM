"""Lightweight offline checks for the data and response contract."""
import json
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from prepare_data import convert_row
from response_schema import is_valid_response, normalize_response, resolve_placeholders


class PipelineChecks(unittest.TestCase):
    def test_conversion_uses_only_dataset_labels(self):
        row = {"instruction": "Where is my order?", "category": "ORDER", "intent": "track_order",
               "response": "Please share your order reference."}
        record = convert_row(row)
        self.assertEqual([m["role"] for m in record["messages"]], ["user", "assistant"])
        target = json.loads(record["messages"][1]["content"])
        self.assertEqual(set(target), {"category", "intent", "response"})
        self.assertEqual(target["intent"], "track_order")

    def test_unknown_dataset_label_is_rejected(self):
        result = {"category": "DELIVERY", "intent": "delivery_issue", "response": "Please share details."}
        self.assertFalse(is_valid_response(result, {"DELIVERY"}, {"delivery_period"}))

    def test_valid_dataset_labels_are_accepted(self):
        result = {"category": "DELIVERY", "intent": "delivery_period", "response": "Please share details."}
        self.assertTrue(is_valid_response(result, {"DELIVERY"}, {"delivery_period"}))

    def test_delivery_status_alias_is_normalized(self):
        result = {"category": "DELIVERY", "intent": "delivery_status", "response": "Please share details."}
        normalized = normalize_response(result, {"DELIVERY"}, {"track_order"})
        self.assertEqual(normalized["intent"], "track_order")
        self.assertTrue(is_valid_response(normalized, {"DELIVERY"}, {"track_order"}))

    def test_order_number_template_uses_customer_provided_number(self):
        result = {"category": "DELIVERY", "intent": "track_order",
                  "response": "Your order {{Order Number}} is being reviewed."}
        resolved = resolve_placeholders(result, "My order number is 12345")
        self.assertIn("order number 12345", resolved["response"])
        self.assertNotIn("{{", resolved["response"])

    def test_unresolved_template_is_not_shown_to_customer(self):
        result = {"category": "DELIVERY", "intent": "track_order",
                  "response": "Your order {{Order Number}} is being reviewed."}
        resolved = resolve_placeholders(result, "My package is late")
        self.assertNotIn("{{", resolved["response"])

    def test_order_timing_is_not_mistaken_for_order_number(self):
        result = {"category": "DELIVERY", "intent": "track_order",
                  "response": "Your order {{Order Number}} is being reviewed."}
        resolved = resolve_placeholders(result, "I placed an order yesterday")
        self.assertNotIn("yesterday", resolved["response"])
        self.assertNotIn("{{", resolved["response"])

    def test_tracking_reply_does_not_claim_email_or_order_access(self):
        result = {"category": "DELIVERY", "intent": "track_order",
                  "response": "I'll send you a confirmation email shortly."}
        resolved = resolve_placeholders(result, "My order number is 12345")
        self.assertIn("order number 12345", resolved["response"])
        self.assertIn("cannot access live tracking", resolved["response"])
        self.assertNotIn("send", resolved["response"].lower())

    def test_test_cases_file_is_valid_jsonl_if_populated(self):
        path = ROOT / "data" / "test_cases.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                case = json.loads(line)
                self.assertIn("message", case)
                self.assertIn("expected_intent", case)


if __name__ == "__main__":
    unittest.main()
