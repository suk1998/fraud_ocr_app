# ai_agent.py
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
