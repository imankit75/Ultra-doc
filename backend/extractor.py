"""
Structured extraction module.
Extracts shipment / pick-list / order data fields from a logistics document
using Gemma 3 4B (local GGUF).
"""

import json
import re

from backend.config import config
from backend.rag_engine import call_llm
from backend.document_processor import get_full_text


EXTRACTION_SYSTEM_PROMPT = """You are a logistics document data extraction specialist.
Your job is to extract ALL structured data from the provided document text.

You MUST return ONLY valid JSON with exactly these fields:
{
    "document_type": "string — e.g. Pick List, Invoice, Bill of Lading, Purchase Order, or null",
    "document_number": "string or null",
    "date": "string (ISO format YYYY-MM-DD) or null",
    "shipper": "string (company / sender name and address) or null",
    "consignee": "string (receiver name and address) or null",
    "consignee_phone": "string or null",
    "consignee_email": "string or null",
    "storage_location": "string (warehouse / storage address) or null",
    "carrier_name": "string or null",
    "shipment_id": "string or null",
    "pickup_datetime": "string (ISO format) or null",
    "delivery_datetime": "string (ISO format) or null",
    "equipment_type": "string or null",
    "mode": "string or null",
    "rate": "number or null",
    "currency": "string (e.g. USD) or null",
    "weight": "string (include unit) or null",
    "line_items": [
        {
            "sku": "string or null",
            "description": "string",
            "quantity": "number or null",
            "unit_price": "number or null",
            "total_price": "number or null"
        }
    ],
    "total_items": "number — sum of all quantities, or null",
    "notes": "string — any additional relevant info, or null"
}

RULES:
1. Extract ONLY from the provided text. Do NOT fabricate data.
2. Use null for any field that cannot be found in the document.
3. For dates, convert to ISO 8601 format if possible (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS).
4. For rate, extract the numeric value only (no currency symbol).
5. For line_items, extract EVERY item row you find. If there are no line items, use an empty list [].
6. Compute total_items by summing all quantities in line_items if possible.
7. Return ONLY the JSON object, no markdown formatting, no explanation."""

EXTRACTION_USER_TEMPLATE = """Extract ALL structured data from the following document:

---
{document_text}
---

Return ONLY a valid JSON object with the required fields."""


def extract_shipment_data(file_path: str) -> dict:
    """
    Extract structured shipment/order data from a document.
    Returns a dict with document fields (nulls for missing).
    """
    # Get full document text
    full_text = get_full_text(file_path)

    if not full_text.strip():
        return _empty_result("Document is empty or could not be parsed.")

    # Truncate if very long (to fit within LLM context window)
    max_chars = 12000
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars] + "\n\n[...document truncated...]"

    user_prompt = EXTRACTION_USER_TEMPLATE.format(document_text=full_text)

    try:
        raw_response = call_llm(
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
    except Exception as e:
        return _empty_result(f"LLM call failed: {str(e)}")

    # Parse JSON from response
    return _parse_extraction_response(raw_response)


def _parse_extraction_response(raw: str) -> dict:
    """Parse the LLM response into a structured dict, handling edge cases."""
    # Try direct JSON parse
    try:
        data = json.loads(raw, strict=False)
        return _normalize_fields(data)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code block
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1), strict=False)
            return _normalize_fields(data)
        except json.JSONDecodeError:
            pass

    # Try to find any JSON-like object in the response
    brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace_match:
        try:
            data = json.loads(brace_match.group(0), strict=False)
            return _normalize_fields(data)
        except json.JSONDecodeError:
            pass

    return _empty_result(f"Could not parse as JSON. Length: {len(raw)}. Tail: {raw[-150:]}")


EXPECTED_FIELDS = [
    "document_type", "document_number", "date",
    "shipper", "consignee", "consignee_phone", "consignee_email",
    "storage_location", "carrier_name",
    "shipment_id", "pickup_datetime", "delivery_datetime",
    "equipment_type", "mode", "rate", "currency", "weight",
    "line_items", "total_items", "notes",
]


def _normalize_fields(data: dict) -> dict:
    """Ensure all expected fields are present, fill missing with null."""
    result = {}
    for field in EXPECTED_FIELDS:
        value = data.get(field, None)
        if field == "line_items":
            if not isinstance(value, list):
                value = []
            else:
                # Clean newlines inside line item string fields too
                cleaned_items = []
                for item in value:
                    if isinstance(item, dict):
                        item = {k: _clean_str(v) for k, v in item.items()}
                    cleaned_items.append(item)
                value = cleaned_items
        elif isinstance(value, str):
            value = _clean_str(value)
        result[field] = value
    return result


def _clean_str(value) -> str:
    """Replace newlines and extra whitespace in a string value."""
    if not isinstance(value, str):
        return value
    # Replace newlines with ', ' and collapse multiple spaces
    cleaned = value.replace("\n", ", ").replace("\r", "")
    # Collapse multiple spaces/commas
    cleaned = re.sub(r',\s*,', ',', cleaned)   # remove double commas
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)  # collapse multiple spaces
    return cleaned.strip(", ").strip()


def _empty_result(error_msg: str = "") -> dict:
    """Return an empty extraction result with all nulls."""
    result = {field: None for field in EXPECTED_FIELDS}
    result["line_items"] = []
    if error_msg:
        result["_error"] = error_msg
    return result
