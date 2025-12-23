# app.py
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

try:
    import pdfplumber
except Exception:
    pdfplumber = None

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

    st.divider()
    st.header("Schema Templates")
    if st.button("Load: Wire/Receipt template"):
        st.session_state["schema"] = DEFAULT_WIRE_SCHEMA.copy()
    if st.button("Load: Etherscan template"):
        st.session_state["schema"] = DEFAULT_ETHERSCAN_SCHEMA.copy()

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
