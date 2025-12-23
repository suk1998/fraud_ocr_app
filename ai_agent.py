# ai_agent.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# OpenAI SDK (recommended: openai>=1.x)
# pip install openai
try:
    from openai import OpenAI
except Exception as e:  # pragma: no cover
    OpenAI = None


@dataclass
class FieldSpec:
    key: str
    type: str = "text"          # text|date|amount|address|hash
    required: bool = False
    hint: str = ""
    regex: str = ""             # optional validation regex


def _compact_schema(schema: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for f in schema:
        key = str(f.get("key", "")).strip()
        if not key:
            continue
        out.append({
            "key": key,
            "type": str(f.get("type", "text")).strip() or "text",
            "required": bool(f.get("required", False)),
            "hint": str(f.get("hint", "")).strip(),
            "regex": str(f.get("regex", "")).strip(),
        })
    return out


def build_prompt(raw_text: str, schema: List[Dict[str, Any]]) -> str:
    schema = _compact_schema(schema)
    fields = "\n".join(
        [
            f"- {f['key']} (type={f['type']}, required={f['required']}): {f.get('hint','')}".rstrip()
            for f in schema
        ]
    )

    # NOTE: We explicitly force JSON-only output and "no hallucination".
    prompt = f"""
You are a strict information extraction engine.

Task:
Extract the following fields from the OCR_TEXT and return JSON only.

Rules:
- Output must be valid JSON and nothing else.
- If a value is not found, set value=null and evidence_text=null.
- Do not guess or hallucinate.
- evidence_text must be copied verbatim from OCR_TEXT.
- confidence is a number from 0.0 to 1.0.

FIELDS:
{fields}

OCR_TEXT:
\"\"\"{raw_text}\"\"\"

OUTPUT_JSON_FORMAT:
{{
  "document_type": "wire|receipt|etherscan|unknown",
  "fields": {{
    "<key>": {{
      "value": <string|null>,
      "evidence_text": <string|null>,
      "confidence": <number>,
      "notes": <string|null>
    }}
  }}
}}
""".strip()
    return prompt


def _safe_json_loads(s: str) -> Dict[str, Any]:
    """
    Attempts to parse JSON even if the model wrapped it with extra text.
    We still demand JSON-only, but this makes the pipeline resilient.
    """
    s = s.strip()
    # If it's already pure JSON
    if s.startswith("{") and s.endswith("}"):
        return json.loads(s)

    # Try to find the first JSON object in the text
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        raise ValueError("Model output did not contain a JSON object.")
    return json.loads(m.group(0))


def extract_with_openai(
    raw_text: str,
    schema: List[Dict[str, Any]],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 1200,
) -> Dict[str, Any]:
    if OpenAI is None:
        raise RuntimeError("openai SDK not installed. Run: pip install openai")

    client = OpenAI()
    prompt = build_prompt(raw_text, schema)

    # Preferred: Chat Completions with response_format json_object (broadly supported)
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You output JSON only."},
                {"role": "user", "content": prompt},
            ],
        )
        content = resp.choices[0].message.content or "{}"
        return _safe_json_loads(content)

    except Exception:
        # Fallback: older/other SDK behavior – no response_format
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": "Output JSON only. No markdown."},
                {"role": "user", "content": prompt},
            ],
        )
        content = resp.choices[0].message.content or "{}"
        return _safe_json_loads(content)


# ---------- Optional: deterministic validation helpers ----------

_ETH_ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")
_TX_HASH_RE = re.compile(r"^0x[a-fA-F0-9]{64}$")

def validate_value(field_type: str, value: Optional[str], regex: str = "") -> bool:
    if value is None:
        return True
    v = str(value).strip()

    if regex:
        try:
            return re.search(regex, v) is not None
        except re.error:
            # bad regex supplied by user; ignore validation
            return True

    if field_type == "address":
        return bool(_ETH_ADDRESS_RE.match(v))
    if field_type == "hash":
        return bool(_TX_HASH_RE.match(v))
    if field_type == "amount":
        # allow "$100,000.00", "19.1532 ETH", etc. (basic)
        return bool(re.search(r"\d", v))
    if field_type == "date":
        # accept many; app can normalize later
        return bool(re.search(r"\d{4}|\d{1,2}[/-]\d{1,2}", v))
    return True
