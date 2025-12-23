# ai_agent.py
<<<<<<< HEAD
# ChatGPT-as-Agent module (JSON-only extraction)
# Uses OpenAI Responses API (OpenAI Python SDK v1+)

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


SCHEMA_KEYS = [
    "DocumentType",
    "SenderName",
    "RecipientName",
    "RecipientBank",
    "TransferDate",
    "TransferAmount",
    "Fee",
    "Asset",
    "WalletAddress",
    "TxID",
    "Network",
    "confidence",
]


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Best-effort extraction of the first JSON object found in model output."""
    if not text:
        return {}

    t = text.strip()

    # Fast path: whole response is JSON
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Find first {...} block
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}

    candidate = m.group(0).strip()
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return {}

    return {}


def _normalize_value(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        if s.lower() in ("n/a", "na", "none", "null", "unknown"):
            return None
        return s
    return str(v).strip() or None


def _normalize_agent_result(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {k: None for k in SCHEMA_KEYS}

    for k in out.keys():
        if k in d:
            out[k] = d.get(k)

    for k in SCHEMA_KEYS:
        if k == "confidence":
            continue
        out[k] = _normalize_value(out[k])

    c = out.get("confidence")
    try:
        cf = float(c) if c is not None else 0.0
    except Exception:
        cf = 0.0
    cf = max(0.0, min(1.0, cf))
    out["confidence"] = cf

    return out


def build_prompt(clean_text: str, raw_text: str = "") -> str:
    """
    Extraction prompt:
    - Force strict JSON contract
    - Explicitly instruct to look around labels even with OCR spacing noise
    """
    # Keep it short but operational
    return (
        "You are an information extraction agent.\n"
        "Return ONE JSON object only.\n"
        "\n"
        "Keys (must exist):\n"
        "DocumentType, SenderName, RecipientName, RecipientBank, TransferDate, TransferAmount, Fee, Asset, WalletAddress, TxID, Network, confidence\n"
        "\n"
        "DocumentType must be one of: CHASE_WIRE, BINANCE_WITHDRAWAL, OTHER\n"
        "\n"
        "Rules:\n"
        "- JSON only. No prose.\n"
        "- If unknown, use null.\n"
        "- Do NOT invent. Use the text only.\n"
        "- Prefer exact substrings.\n"
        "- For fields like Sender/Recipient/Bank, look near labels like:\n"
        "  SENDER, RECIPIENT, RECIPIENT BANK, Wire Transfer Date, Transfer Amount, Transfer Fees, Txid, Address, Network.\n"
        "- OCR may insert spaces (e.g., S E N D E R) or miss punctuation; still infer using nearby lines.\n"
        "\n"
        "CLEAN_TEXT:\n"
        f"{clean_text}\n"
        "\n"
        "RAW_TEXT (may contain more line breaks):\n"
        f"{raw_text}\n"
    )


def extract_fields_with_agent(
    clean_text: str,
    raw_text: str = "",
    model: str = "gpt-5",
    api_key: Optional[str] = None,
    timeout_s: int = 60,
) -> Dict[str, Any]:
    """
    Calls OpenAI Responses API and returns normalized extraction dict.
    If OpenAI SDK is not installed or API key is missing, returns empty dict.
    """
    if not clean_text or not clean_text.strip():
        return {}

    if OpenAI is None:
        return {}

    key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not key:
        return {}

    client = OpenAI(api_key=key)
    prompt = build_prompt(clean_text=clean_text, raw_text=raw_text)

    resp = client.responses.create(
        model=model,
        input=prompt,
        timeout=timeout_s,
    )

    out_text = getattr(resp, "output_text", "") or ""
    raw_obj = _extract_json_object(out_text)
    return _normalize_agent_result(raw_obj)
=======
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
>>>>>>> 3429900 (Remove temporary app file and add AI-driven OCR extractor)
