# app.py
<<<<<<< HEAD
# Fraud OCR Extractor: OCR -> Text Clean -> Rule Extraction (+ Optional AI Agent) -> Excel/CSV + Readable TXT
# 2025-12-20

import os
import io
import zipfile
import platform
import shutil
import re
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
from PIL import Image, ImageOps, ImageFilter

import pytesseract
import pdfplumber
from pdf2image import convert_from_bytes

# Optional AI agent import (keep app runnable even if file is missing)
try:
    from ai_agent import extract_fields_with_agent
except Exception:
    extract_fields_with_agent = None


# -----------------------------
# Tesseract auto-detection
# -----------------------------
def _auto_set_tesseract():
    """Auto-detect Tesseract path on Windows/Linux."""
    if platform.system() == "Windows":
        for c in (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ):
            if os.path.exists(c):
                pytesseract.pytesseract.tesseract_cmd = c
                return
    else:
        path = shutil.which("tesseract")
        if path:
            pytesseract.pytesseract.tesseract_cmd = path


_auto_set_tesseract()
=======
from __future__ import annotations

import io
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from PIL import Image

# OCR
try:
    import pytesseract
except Exception:
    pytesseract = None
>>>>>>> 3429900 (Remove temporary app file and add AI-driven OCR extractor)

try:
    import pdfplumber
except Exception:
    pdfplumber = None

<<<<<<< HEAD

# -----------------------------
# OCR helpers
# -----------------------------
def _preprocess(img: Image.Image) -> Image.Image:
    """Preprocess image to improve OCR accuracy."""
    g = ImageOps.grayscale(img)
    g = ImageOps.autocontrast(g)
    g = g.filter(ImageFilter.SHARPEN)
    g = g.point(lambda p: 255 if p > 200 else (0 if p < 135 else p))
    return g


def ocr_pil(img: Image.Image, lang: str = "eng+kor") -> str:
    """Run OCR on a PIL image."""
    return pytesseract.image_to_string(_preprocess(img), lang=lang, config="--oem 3 --psm 6")


def ocr_image_bytes(b: bytes) -> str:
    """OCR for image bytes."""
    with Image.open(io.BytesIO(b)) as im:
        return ocr_pil(im)


def pdf_text_or_ocr_bytes(b: bytes) -> str:
    """Extract text from a PDF; if not text-based, OCR each page."""
    # 1) Direct text extraction (text-based PDFs)
    try:
        t_all, n = [], 0
        with pdfplumber.open(io.BytesIO(b)) as pdf:
            for p in pdf.pages:
                t = p.extract_text() or ""
                t_all.append(t)
                n += len(t)
        if n >= 50:
            return "\n".join(t_all)
    except Exception:
        pass

    # 2) OCR fallback (scanned PDFs)
    texts = []
    for page_img in convert_from_bytes(b, dpi=300):
        texts.append(ocr_pil(page_img))
    return "\n".join(texts)


def do_ocr_any(name: str, content: bytes) -> dict:
    """Dispatch OCR based on file extension."""
    ext = Path(name).suffix.lower()
    try:
        if ext in IMG_EXT:
            raw = ocr_image_bytes(content)
        elif ext in PDF_EXT:
            raw = pdf_text_or_ocr_bytes(content)
        else:
            return {"filename": name, "raw_text": f"Unsupported file type: {ext}"}
    except Exception as e:
        return {"filename": name, "raw_text": f"⚠️ OCR error: {e}"}

    return {"filename": name, "raw_text": (raw or "").strip()}


# -----------------------------
# Text cleaning
# -----------------------------
DEFAULT_REPLACEMENTS = [
    ("Ox", "0x"),
    ("O x", "0x"),
    ("0 x", "0x"),
    ("—", "-"),
    ("–", "-"),
    ("\u200b", ""),
]


def clean_ocr_text(t: str, collapse_ws: bool = True) -> str:
    """
    Cleaning for extraction:
    - Keep line breaks as much as possible (labels rely on structure)
    - Normalize common OCR artifacts
    """
    if not t:
        return ""

    s = t
    for a, b in DEFAULT_REPLACEMENTS:
        s = s.replace(a, b)

    # Remove some decorative symbols often produced by OCR
    s = re.sub(r"[©®™]", " ", s)

    if collapse_ws:
        # Keep newlines, only collapse excessive spaces
        s = re.sub(r"[ \t]+", " ", s)
        # Normalize CRLF
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        # Reduce too many blank lines
        s = re.sub(r"\n{3,}", "\n\n", s)

    return s.strip()


def clean_text_for_txt(t: str) -> str:
    """Cleaning for readable TXT output (more aggressive)."""
    if not t:
        return ""

    s = t
    for a, b in DEFAULT_REPLACEMENTS:
        s = s.replace(a, b)

    s = re.sub(r"[©®™@•■◆●◼︎]", " ", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)

    return s.strip()


# -----------------------------
# Robust label/block extraction (Rule-based)
# -----------------------------
def spaced_label_regex(label: str) -> str:
    """Match labels even if OCR inserts spaces between letters."""
    parts = label.strip().split()

    def word_pat(w: str) -> str:
        return r"\s*".join(map(re.escape, w))

    return r"\s+".join(word_pat(p) for p in parts)


def _block_after_label(text: str, label: str, max_chars: int = 400) -> str:
    """
    Capture a block right after a label, either:
    - Same line: 'SENDER: JOHN DOE'
    - Next lines:
        SENDER:
        JOHN DOE
        123 STREET...
    Stop when another ALL-CAPS label-like line starts.
    """
    lab = spaced_label_regex(label)

    # 1) Same-line capture
    m = re.search(rf"(?im)^[ \t]*{lab}\s*[:\-]?\s*([^\n\r]{{2,{max_chars}}})$", text)
    if m:
        return (m.group(1) or "").strip()

    # 2) Multi-line block capture
    m = re.search(rf"(?im)^[ \t]*{lab}\s*[:\-]?\s*(?:\r?\n)+", text)
    if not m:
        return ""

    start = m.end()
    tail = text[start : start + 2000]

    # Stop conditions: a new label line (heuristic)
    # Example: 'RECIPIENT:' or 'RECIPIENT BANK:' or 'Amount to be ...'
    stop = re.search(r"(?m)^\s*[A-Z][A-Z0-9 /&().'-]{2,40}\s*:\s*$", tail)
    if stop:
        tail = tail[: stop.start()]

    return tail[:max_chars].strip()


def _pick_name_from_block(block: str) -> str | None:
    """
    Choose a plausible 'name/company' line from a block.
    Avoid dates, phone numbers, pure IDs, and address-like lines.
    """
    if not block:
        return None

    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    if not lines:
        # If it was same-line content without newlines
        lines = [block.strip()]

    def bad_line(ln: str) -> bool:
        u = ln.upper()
        if re.search(r"\b\d{4}-\d{2}-\d{2}\b", ln):
            return True
        if re.search(r"\b[A-Za-z]+\s+\d{1,2},\s*\d{4}\b", ln):
            return True
        if re.search(r"\b\d{3}[- ]?\d{3}[- ]?\d{4}\b", ln):
            return True
        if re.fullmatch(r"[\d\s\-/.,]+", ln):
            return True
        if "HTTP" in u or "WWW." in u:
            return True
        if "USD" in u or "USDT" in u:
            return True
        return False

    def looks_like_address(ln: str) -> bool:
        # Very rough address heuristics
        if re.search(r"\b(ST|STREET|AVE|AVENUE|RD|ROAD|BLVD|DR|WAY|HWY)\b", ln.upper()):
            return True
        if re.search(r"\b\d{2,6}\b", ln) and re.search(r"[A-Za-z]", ln):
            return True
        return False

    for ln in lines:
        if bad_line(ln):
            continue
        if looks_like_address(ln):
            continue
        # Must contain letters
        if not re.search(r"[A-Za-z]", ln):
            continue
        # Keep it within a reasonable length
        if 2 <= len(ln) <= 120:
            return ln.strip()

    return None


def detect_doc_type(text: str) -> str:
    u = text.upper() if text else ""
    if "CHASE" in u and "WIRE" in u:
        return "CHASE_WIRE"
    if "BINANCE" in u or "USDT" in u or "TXID" in u or "WITHDRAWAL" in u:
        return "BINANCE_WITHDRAWAL"
    return "OTHER"


def extract_financial_fields_rule(clean_text: str) -> dict:
    """Rule-based extraction from cleaned text."""
    fields = {
        "DocumentType": detect_doc_type(clean_text),
        "SenderName": None,
        "RecipientName": None,
        "RecipientBank": None,
        "TransferDate": None,
        "TransferAmount": None,
        "Fee": None,
        "Asset": None,
        "WalletAddress": None,
        "TxID": None,
        "Network": None,
    }
    if not clean_text:
        return fields

    text = clean_text
    u = text.upper()

    # -----------------------------
    # Chase wire
    # -----------------------------
    if "CHASE" in u and "WIRE" in u:
        sender_block = _block_after_label(text, "SENDER")
        fields["SenderName"] = _pick_name_from_block(sender_block)

        # Recipient label variants (OCR misspells)
        for lbl in ["RECIPIENT", "RECIPENT", "RECEPIENT", "RECIPlENT"]:
            rec_block = _block_after_label(text, lbl)
            v = _pick_name_from_block(rec_block)
            if v:
                fields["RecipientName"] = v
                break

        for lbl in ["RECIPIENT BANK", "RECIPENT BANK", "RECEPIENT BANK"]:
            bank_block = _block_after_label(text, lbl, max_chars=600)
            v = _pick_name_from_block(bank_block) or (bank_block.splitlines()[0].strip() if bank_block else None)
            if v:
                fields["RecipientBank"] = v
                break

        # Dates (wire transfer date / today's date)
        m = re.search(
            r"(?i)(Wire\s*Transfer\s*Date|Today'?s\s*Date)\s*[:\-]?\s*([A-Za-z]+\s+\d{1,2},\s*\d{4})",
            text,
        )
        if m:
            fields["TransferDate"] = m.group(2).strip()

        # Amount / fees
        m = re.search(
            r"(?i)Transfer\s*Amount\s*[:\-]?\s*\$?\s*([\d]{1,3}(?:,\d{3})*(?:\.\d{2})?\s*[A-Z]{3})",
            text,
        )
        if m:
            fields["TransferAmount"] = m.group(1).strip()

        m = re.search(
            r"(?i)(Transfer\s*Fees|Other\s*Fees)\s*[:\-]?\s*\+?\$?\s*([\d]{1,3}(?:,\d{3})*(?:\.\d{2})?\s*[A-Z]{3})",
            text,
        )
        if m:
            fields["Fee"] = m.group(2).strip()

    # -----------------------------
    # Binance withdrawal
    # -----------------------------
    if ("BINANCE" in u) or ("USDT" in u) or ("TXID" in u) or ("WITHDRAWAL" in u):
        m = re.search(r"(?is)\bAmount\b.*?\n\s*([\d,]+\s*[A-Z]+)", text)
        if m:
            fields["Asset"] = m.group(1).strip()

        m = re.search(r"(?is)\bNetwork\b.*?\n\s*([A-Z0-9\-]+)", text)
        if m:
            fields["Network"] = m.group(1).strip()

        m = re.search(r"(?is)\bAddress\b.*?\n\s*([A-Za-z0-9]+)", text)
        if m:
            fields["WalletAddress"] = m.group(1).strip()

        m = re.search(r"(?is)\bTxid\b.*?\n\s*([A-Fa-f0-9]{20,})", text)
        if m:
            fields["TxID"] = m.group(1).strip()

        if not fields["TransferDate"]:
            m = re.search(r"\b(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\b", text)
            if m:
                fields["TransferDate"] = m.group(1).strip()

        if not fields["Fee"]:
            m = re.search(r"(?is)\bNetwork\s*fee\b.*?\n\s*([\d.,]+\s*[A-Z]+)", text)
            if m:
                fields["Fee"] = m.group(1).strip()

    # Generic EVM catches
    if not fields["WalletAddress"]:
        m = re.search(r"\b0x[a-fA-F0-9]{20,}\b", text)
        if m:
            fields["WalletAddress"] = m.group(0)

    if not fields["TxID"]:
        m = re.search(r"\b0x[a-fA-F0-9]{64}\b", text)
        if m:
            fields["TxID"] = m.group(0)

    return fields


# -----------------------------
# Filename sanitization
# -----------------------------
def sanitize_basename(s: str) -> str:
    """Sanitize a base filename (no extension)."""
    s = (s or "").strip()
    if not s:
        s = "ocr_results"
    s = re.sub(r'[\\/:*?"<>|]+', "_", s)
    return s[:120]


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Fraud OCR Extractor", layout="wide")
st.title("Fraud OCR Extractor (Images + PDF + ZIP)")
st.caption("OCR -> Clean -> (Rule + Optional AI Agent) -> Export to Excel/CSV + Readable TXT.")

tab1, tab2 = st.tabs(["Upload Files", "Upload Folder (ZIP)"])

with st.sidebar:
    st.subheader("Export Options")
    base_input = st.text_input("Base file name", value="ocr_results")
    add_ts = st.checkbox("Append timestamp (YYYYMMDD_HHMMSS)", value=False)
    base_name = sanitize_basename(base_input)
    if add_ts:
        base_name = base_name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    st.divider()

    st.subheader("Extraction Options")
    collapse_ws = st.checkbox("Collapse whitespace/newlines (keep structure)", value=True)

    use_ai_agent = st.checkbox("Use AI Agent (ChatGPT)", value=False)
    ai_model = st.text_input("AI model (OpenAI)", value="gpt-5")
    st.caption("Requires OPENAI_API_KEY in your environment. If missing, AI Agent will be skipped.")

    include_text_cols = st.checkbox("Include Raw/Clean text columns in CSV/Excel", value=False)
    show_debug_text = st.checkbox("Show debug text (Raw/Clean) in UI", value=True)
=======
from ai_agent import extract_with_openai, validate_value


# -----------------------------
# Default schemas (templates)
# -----------------------------
DEFAULT_WIRE_SCHEMA = [
    {"key": "wire_transfer_date", "type": "date", "required": True, "hint": "Wire Transfer Date / Wire Date", "regex": ""},
    {"key": "sender", "type": "text", "required": False, "hint": "SENDER / Sender Information", "regex": ""},
    {"key": "sender_information", "type": "text", "required": False, "hint": "Sender information section (name/address/phone if present)", "regex": ""},
    {"key": "chase_wire_tracking_number", "type": "text", "required": False, "hint": "Tracking Number / Ref ID", "regex": ""},
    {"key": "recipient", "type": "text", "required": False, "hint": "RECIPIENT", "regex": ""},
    {"key": "recipient_bank", "type": "text", "required": False, "hint": "RECIPIENT BANK", "regex": ""},
    {"key": "transfer_amount", "type": "amount", "required": False, "hint": "Transfer Amount / Wire Amount", "regex": ""},
    {"key": "fees", "type": "amount", "required": False, "hint": "Fees / Transfer Fee", "regex": ""},
    {"key": "taxes", "type": "amount", "required": False, "hint": "Taxes", "regex": ""},
    {"key": "total", "type": "amount", "required": False, "hint": "Total", "regex": ""},
]

DEFAULT_ETHERSCAN_SCHEMA = [
    {"key": "transaction_action", "type": "text", "required": False, "hint": "TRANSACTION ACTION line (e.g., Transfer X ETH to ...)", "regex": ""},
    {"key": "to_address", "type": "address", "required": True, "hint": "to 0x... address", "regex": r"0x[a-fA-F0-9]{40}"},
    {"key": "from_address", "type": "address", "required": False, "hint": "from 0x... address (if present)", "regex": r"0x[a-fA-F0-9]{40}"},
    {"key": "tx_hash", "type": "hash", "required": True, "hint": "Transaction Hash: 0x...", "regex": r"0x[a-fA-F0-9]{64}"},
    {"key": "status", "type": "text", "required": False, "hint": "Status: Success/Fail", "regex": ""},
    {"key": "block", "type": "text", "required": False, "hint": "Block number", "regex": r"^\d+$"},
    {"key": "timestamp_utc", "type": "date", "required": False, "hint": "Timestamp (UTC)", "regex": ""},
    {"key": "value_eth", "type": "amount", "required": False, "hint": "ETH value", "regex": ""},
    {"key": "value_usd", "type": "amount", "required": False, "hint": "USD value", "regex": ""},
]


# -----------------------------
# OCR utilities
# -----------------------------
def ocr_image_pytesseract(img: Image.Image) -> str:
    if pytesseract is None:
        raise RuntimeError("pytesseract is not installed. Run: pip install pytesseract")
    # Convert to RGB for safety
    if img.mode != "RGB":
        img = img.convert("RGB")
    text = pytesseract.image_to_string(img)
    return text or ""


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber is not installed. Run: pip install pdfplumber")
    out = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                out.append(t)
            else:
                # no text layer – fallback to OCR on rendered image
                # render page to image via pdfplumber
                pil_img = page.to_image(resolution=200).original
                out.append(ocr_image_pytesseract(pil_img))
    return "\n\n".join(out).strip()


def detect_doc_type_hint(raw_text: str) -> str:
    t = raw_text.lower()
    if "etherscan.io" in t or "transaction hash" in t or "transaction action" in t:
        return "etherscan"
    if "wire transfer" in t or "recipient bank" in t or "sender information" in t or "chase" in t:
        return "wire"
    if "bank of america" in t and "wire transfer receipt" in t:
        return "receipt"
    return "unknown"


# -----------------------------
# Flatten & validation
# -----------------------------
def flatten_result(result: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns:
      row_values: key -> value
      row_meta: key -> meta dict (evidence/confidence/notes)
    """
    fields = result.get("fields", {}) if isinstance(result, dict) else {}
    row_values = {}
    row_meta = {}
    for k, v in fields.items():
        if isinstance(v, dict):
            row_values[k] = v.get("value", None)
            row_meta[k] = {
                "evidence_text": v.get("evidence_text", None),
                "confidence": v.get("confidence", None),
                "notes": v.get("notes", None),
            }
        else:
            row_values[k] = v
            row_meta[k] = {}
    return row_values, row_meta


def build_validation_report(schema: List[Dict[str, Any]], values: Dict[str, Any]) -> List[Dict[str, Any]]:
    report = []
    for f in schema:
        key = str(f.get("key", "")).strip()
        if not key:
            continue
        ftype = str(f.get("type", "text")).strip() or "text"
        regex = str(f.get("regex", "")).strip()
        required = bool(f.get("required", False))
        val = values.get(key, None)

        ok = validate_value(ftype, val, regex=regex)
        missing = required and (val is None or str(val).strip() == "")

        report.append({
            "field": key,
            "type": ftype,
            "required": required,
            "value": val,
            "valid": bool(ok) and not bool(missing),
            "issue": "missing_required" if missing else ("regex_or_type_validation_failed" if not ok else ""),
        })
    return report


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Fraud Evidence OCR Extractor", layout="wide")

st.title("Fraud Evidence OCR Extractor (Schema-driven + AI Agent)")

with st.sidebar:
    st.header("AI Settings")
    model = st.text_input("OpenAI model", value="gpt-4o-mini")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05)
    max_tokens = st.slider("Max tokens", 300, 4000, 1200, 100)
>>>>>>> 3429900 (Remove temporary app file and add AI-driven OCR extractor)

    st.divider()
    st.header("Schema Templates")
    if st.button("Load: Wire/Receipt template"):
        st.session_state["schema"] = DEFAULT_WIRE_SCHEMA.copy()
    if st.button("Load: Etherscan template"):
        st.session_state["schema"] = DEFAULT_ETHERSCAN_SCHEMA.copy()

<<<<<<< HEAD
with tab1:
    up = st.file_uploader(
        "Select multiple image or PDF files",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "pdf"],
        accept_multiple_files=True,
    )
    if up and st.button("Run OCR (Files)"):
        with st.spinner("Processing..."):
            for f in up:
                results.append(do_ocr_any(f.name, f.read()))

with tab2:
    zip_file = st.file_uploader("Upload a folder as a ZIP file", type=["zip"], key="zip")
    if zip_file and st.button("Run OCR (ZIP)"):
        with st.spinner("Extracting and processing..."):
            with zipfile.ZipFile(io.BytesIO(zip_file.read())) as zf:
                for zi in zf.infolist():
                    if zi.is_dir():
                        continue
                    name = zi.filename
                    ext = Path(name).suffix.lower()
                    if ext in (IMG_EXT | PDF_EXT):
                        content = zf.read(zi)
                        results.append(do_ocr_any(Path(name).name, content))

if results:
    rows = []
    debug_payload = []

    for r in results:
        filename = r.get("filename")
        raw = r.get("raw_text", "") or ""
        clean = clean_ocr_text(raw, collapse_ws=collapse_ws)

        rule = extract_financial_fields_rule(clean)

        agent = {}
        agent_conf = None
        if use_ai_agent and extract_fields_with_agent is not None:
            agent = extract_fields_with_agent(clean_text=clean, raw_text=raw, model=ai_model)
            agent_conf = agent.get("confidence")

        # Merge: prefer agent value, fallback to rule
        merged = {}
        for k in rule.keys():
            merged[k] = (agent.get(k) if agent else None) or rule.get(k)

        # Conflict detection (only meaningful if both present)
        conflicts = []
        if agent:
            for k in rule.keys():
                rv = rule.get(k)
                av = agent.get(k)
                if rv and av and str(rv) != str(av):
                    conflicts.append(k)

        row = {
            "Filename": filename,
            **merged,
            "AI_confidence": agent_conf,
            "Conflicts": ";".join(conflicts) if conflicts else "",
        }

        if include_text_cols:
            row["RawText"] = raw
            row["CleanText"] = clean

        rows.append(row)
        debug_payload.append({"Filename": filename, "RawText": raw, "CleanText": clean})

    base_cols = [
        "Filename",
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
        "AI_confidence",
        "Conflicts",
    ]
    if include_text_cols:
        base_cols += ["RawText", "CleanText"]

    df = pd.DataFrame(rows)
    cols = [c for c in base_cols if c in df.columns]
    df = df[cols]

    st.subheader("Preview")
    st.dataframe(df, use_container_width=True, height=560)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Files processed", len(df))
    c2.metric("DocType detected", int((df["DocumentType"] != "OTHER").sum()) if "DocumentType" in df else 0)
    c3.metric("WalletAddress detected", int(df["WalletAddress"].notna().sum()) if "WalletAddress" in df else 0)
    c4.metric("TxID detected", int(df["TxID"].notna().sum()) if "TxID" in df else 0)

    if show_debug_text:
        st.subheader("Debug (Raw/Clean text)")
        st.caption("If Sender/Recipient is missing, check whether OCR output contains the labels and nearby lines.")
        for item in debug_payload:
            with st.expander(f"Text view: {item['Filename']}"):
                st.markdown("**RawText**")
                st.text(item["RawText"][:12000] if item["RawText"] else "")
                st.markdown("**CleanText**")
                st.text(item["CleanText"][:12000] if item["CleanText"] else "")

    # CSV download
    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name=base_name + ".csv",
        mime="text/csv",
    )

    # Excel download
    xlsx_buf = io.BytesIO()
    with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="ocr_results")
    st.download_button(
        "Download Excel",
        data=xlsx_buf.getvalue(),
        file_name=base_name + ".xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # Readable TXT (combined)
    txt_blocks = []
    for r in results:
        fn = r.get("filename")
        raw = r.get("raw_text", "") or ""
        readable = clean_text_for_txt(raw)
        txt_blocks.append("===== " + str(fn) + " =====\n" + readable + "\n")

    st.download_button(
        "Download TXT (readable, combined)",
        data="\n".join(txt_blocks).encode("utf-8-sig"),
        file_name=base_name + "_readable.txt",
        mime="text/plain",
    )

    # Readable TXT per-file ZIP
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        for r in results:
            fn = r.get("filename") or "file"
            stem = sanitize_basename(Path(str(fn)).stem) or "file"
            raw = r.get("raw_text", "") or ""
            readable = clean_text_for_txt(raw)
            z.writestr(stem + ".txt", readable)
    zip_buf.seek(0)

    st.download_button(
        "Download TXT (readable, per-file ZIP)",
        data=zip_buf.getvalue(),
        file_name=base_name + "_readable_texts.zip",
        mime="application/zip",
    )

else:
    st.info("Please upload files or a ZIP folder.")
=======
# init schema
if "schema" not in st.session_state:
    st.session_state["schema"] = DEFAULT_WIRE_SCHEMA.copy()

if "ocr_text" not in st.session_state:
    st.session_state["ocr_text"] = ""

if "result_json" not in st.session_state:
    st.session_state["result_json"] = None

if "result_values" not in st.session_state:
    st.session_state["result_values"] = None

if "result_meta" not in st.session_state:
    st.session_state["result_meta"] = None


tab1, tab2, tab3, tab4 = st.tabs(["1) Upload & OCR", "2) Schema Builder", "3) Extract & Review", "4) Export"])

# 1) Upload & OCR
with tab1:
    colL, colR = st.columns([1, 1])

    with colL:
        st.subheader("Upload files")
        uploaded = st.file_uploader(
            "Upload image/PDF (single file)",
            type=["png", "jpg", "jpeg", "webp", "pdf"],
            accept_multiple_files=False,
        )

        if uploaded is not None:
            file_type = uploaded.type
            data = uploaded.getvalue()

            if file_type == "application/pdf" or uploaded.name.lower().endswith(".pdf"):
                st.info("PDF detected. Extracting text (pdfplumber) and OCR fallback if needed.")
                try:
                    text = extract_text_from_pdf(data)
                    st.session_state["ocr_text"] = text
                    st.success("PDF text extraction complete.")
                except Exception as e:
                    st.error(f"PDF extraction failed: {e}")

            else:
                st.info("Image detected. Running OCR (pytesseract).")
                try:
                    img = Image.open(io.BytesIO(data))
                    st.image(img, caption=uploaded.name, use_container_width=True)
                    text = ocr_image_pytesseract(img)
                    st.session_state["ocr_text"] = text
                    st.success("Image OCR complete.")
                except Exception as e:
                    st.error(f"Image OCR failed: {e}")

    with colR:
        st.subheader("OCR text")
        st.session_state["ocr_text"] = st.text_area(
            "You can edit OCR text before extraction",
            value=st.session_state["ocr_text"],
            height=500,
        )
        hint = detect_doc_type_hint(st.session_state["ocr_text"])
        st.caption(f"Detected hint: {hint}")

# 2) Schema Builder
with tab2:
    st.subheader("Schema Builder (User-defined columns)")
    st.write("Edit the schema fields. Add rows for new columns you want extracted.")

    schema_df = pd.DataFrame(st.session_state["schema"])
    edited_df = st.data_editor(
        schema_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "key": st.column_config.TextColumn("key (column name)", required=True),
            "type": st.column_config.SelectboxColumn("type", options=["text", "date", "amount", "address", "hash"]),
            "required": st.column_config.CheckboxColumn("required"),
            "hint": st.column_config.TextColumn("hint (labels/keywords)"),
            "regex": st.column_config.TextColumn("regex (optional)"),
        },
    )

    st.session_state["schema"] = edited_df.to_dict(orient="records")

    st.caption("Tip: For crypto addresses/hashes, set type=address/hash and add regex if you want strict validation.")

# 3) Extract & Review
with tab3:
    st.subheader("AI Extraction")
    colA, colB = st.columns([1, 1])

    with colA:
        st.write("Current schema keys:")
        keys = [str(x.get("key", "")).strip() for x in st.session_state["schema"] if str(x.get("key", "")).strip()]
        st.code("\n".join(keys) if keys else "(no keys)")

        if st.button("Run AI extraction", type="primary"):
            raw_text = st.session_state["ocr_text"].strip()
            if not raw_text:
                st.error("OCR text is empty. Upload a file or paste OCR text first.")
            elif not keys:
                st.error("Schema has no valid keys. Add schema fields first.")
            else:
                try:
                    result = extract_with_openai(
                        raw_text=raw_text,
                        schema=st.session_state["schema"],
                        model=model,
                        temperature=float(temperature),
                        max_tokens=int(max_tokens),
                    )
                    st.session_state["result_json"] = result

                    values, meta = flatten_result(result)
                    st.session_state["result_values"] = values
                    st.session_state["result_meta"] = meta

                    st.success("Extraction complete.")
                except Exception as e:
                    st.error(f"AI extraction failed: {e}")

    with colB:
        st.write("Result JSON (raw):")
        if st.session_state["result_json"] is not None:
            st.json(st.session_state["result_json"])
        else:
            st.info("No extraction result yet.")

    st.divider()
    st.subheader("Review table (values)")

    if st.session_state["result_values"] is not None:
        values = st.session_state["result_values"]
        meta = st.session_state["result_meta"] or {}

        # Value table
        df_values = pd.DataFrame([values])
        st.dataframe(df_values, use_container_width=True)

        # Evidence/confidence table
        st.subheader("Evidence / Confidence")
        rows = []
        for k, v in values.items():
            m = meta.get(k, {})
            rows.append({
                "field": k,
                "value": v,
                "confidence": m.get("confidence", None),
                "evidence_text": m.get("evidence_text", None),
                "notes": m.get("notes", None),
            })
        df_meta = pd.DataFrame(rows)
        st.dataframe(df_meta, use_container_width=True)

        # Validation report
        st.subheader("Validation report")
        report = build_validation_report(st.session_state["schema"], values)
        df_report = pd.DataFrame(report)
        st.dataframe(df_report, use_container_width=True)

        invalid = df_report[df_report["valid"] == False]
        if len(invalid) > 0:
            st.warning("Some fields failed validation or required fields are missing. Review evidence_text and/or adjust schema/regex.")
    else:
        st.info("Run AI extraction first.")

# 4) Export
with tab4:
    st.subheader("Export")
    if st.session_state["result_values"] is None:
        st.info("Nothing to export yet. Run extraction first.")
    else:
        values = st.session_state["result_values"]
        meta = st.session_state["result_meta"] or {}
        result_json = st.session_state["result_json"] or {}

        df = pd.DataFrame([values])

        col1, col2, col3 = st.columns(3)

        with col1:
            csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("Download CSV", data=csv_bytes, file_name="extracted.csv", mime="text/csv")

        with col2:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="values")
                pd.DataFrame([
                    {"field": k, **(meta.get(k, {}))}
                    for k in values.keys()
                ]).to_excel(writer, index=False, sheet_name="meta")
                pd.DataFrame([{"json": json.dumps(result_json, ensure_ascii=False)}]).to_excel(
                    writer, index=False, sheet_name="raw_json"
                )
            st.download_button(
                "Download XLSX",
                data=buf.getvalue(),
                file_name="extracted.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        with col3:
            json_bytes = json.dumps(result_json, ensure_ascii=False, indent=2).encode("utf-8")
            st.download_button("Download JSON", data=json_bytes, file_name="extracted.json", mime="application/json")
>>>>>>> 3429900 (Remove temporary app file and add AI-driven OCR extractor)
