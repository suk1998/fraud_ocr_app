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
    """
    Best-effort extraction of the first JSON object found in the model output.
    We intentionally avoid brittle assumptions (models sometimes wrap JSON with text).
    """
    if not text:
        return {}

    # Fast path: if the whole text is valid JSON
    t = text.strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Find the first {...} block (non-greedy) and attempt to parse
    # This is not perfect but works well in practice for "JSON only" prompts.
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

    # Normalize strings
    for k in SCHEMA_KEYS:
        if k == "confidence":
            continue
        out[k] = _normalize_value(out[k])

    # Normalize confidence
    c = out.get("confidence")
    try:
        cf = float(c) if c is not None else 0.0
    except Exception:
        cf = 0.0
    cf = max(0.0, min(1.0, cf))
    out["confidence"] = cf

    # Light canonicalization
    if out.get("WalletAddress"):
        out["WalletAddress"] = out["WalletAddress"].replace("O", "0") if out["WalletAddress"].startswith("0x") else out["WalletAddress"]
        out["WalletAddress"] = out["WalletAddress"].strip()
    if out.get("TxID"):
        out["TxID"] = out["TxID"].strip()

    return out


def build_prompt(clean_text: str) -> str:
    """
    Agent prompt: force a strict JSON contract.
    Keep it short and operational to reduce hallucination.
    """
    return (
        "You are an information extraction agent.\n"
        "Extract ONLY the following fields from the text and return ONE JSON object.\n"
        "\n"
        "Keys (must exist in JSON):\n"
        "- DocumentType: one of CHASE_WIRE, BINANCE_WITHDRAWAL, OTHER\n"
        "- SenderName\n"
        "- RecipientName\n"
        "- RecipientBank\n"
        "- TransferDate\n"
        "- TransferAmount\n"
        "- Fee\n"
        "- Asset\n"
        "- WalletAddress\n"
        "- TxID\n"
        "- Network\n"
        "- confidence: number between 0.0 and 1.0\n"
        "\n"
        "Rules:\n"
        "- Return JSON only. No prose.\n"
        "- If a field is unknown, use null.\n"
        "- Do not invent values. Use the text only.\n"
        "- Prefer exact substrings from the text.\n"
        "\n"
        "TEXT:\n"
        f"{clean_text}\n"
    )


def extract_fields_with_agent(
    clean_text: str,
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

    prompt = build_prompt(clean_text)

    # Responses API call (recommended for new projects)
    # https://platform.openai.com/docs/api-reference/responses
    resp = client.responses.create(
        model=model,
        input=prompt,
        timeout=timeout_s,
    )

    out_text = getattr(resp, "output_text", "") or ""
    raw_obj = _extract_json_object(out_text)
    norm = _normalize_agent_result(raw_obj)
    return norm
