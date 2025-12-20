# app.py
# Fraud OCR Extractor: OCR -> Clean -> (Rule + Optional AI Agent) -> Excel/CSV + Readable TXT
# 2025-12-19

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

from ai_agent import extract_fields_with_agent


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

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PDF_EXT = {".pdf"}


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
    # 1) Direct text extraction
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

    # 2) OCR fallback for scanned PDFs
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
    """Cleaning for extraction (keep meaning, reduce OCR noise)."""
    if not t:
        return ""

    s = t
    for a, b in DEFAULT_REPLACEMENTS:
        s = s.replace(a, b)

    # Remove some common UI junk but do not over-delete
    s = re.sub(r"[©®™]", " ", s)

    if collapse_ws:
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"\s*\n\s*", "\n", s)
        s = re.sub(r"\n{3,}", "\n\n", s)

    return s.strip()


def clean_text_for_txt(t: str) -> str:
    """Cleaning for readable TXT output (more aggressive)."""
    if not t:
        return ""

    s = t
    for a, b in DEFAULT_REPLACEMENTS:
        s = s.replace(a, b)

    # Remove decorative symbols often produced by OCR
    s = re.sub(r"[©®™@•■◆●◼︎]", " ", s)

    # Normalize whitespace
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)

    # Remove isolated single-letter junk tokens (optional, conservative)
    s = re.sub(r"(?m)^\s*[A-Za-z]\s*$", "", s)

    return s.strip()


# -----------------------------
# Robust label utilities for rule-based extraction
# -----------------------------
def spaced_label_regex(label: str) -> str:
    """Match labels even if OCR inserts spaces between letters."""
    parts = label.strip().split()

    def word_pat(w: str) -> str:
        return r"\s*".join(map(re.escape, w))

    return r"\s+".join(word_pat(p) for p in parts)


def compile_after_label(label: str, tail_regex: str, flags=re.IGNORECASE) -> re.Pattern:
    lab = spaced_label_regex(label)
    pat = lab + tail_regex
    return re.compile(pat, flags)


def compile_line_after_label(label: str) -> re.Pattern:
    """Capture first non-empty line after a label at line start."""
    lab = spaced_label_regex(label)
    pat = r"(?im)^[ \t]*" + lab + r"\s*[:\-]?\s*(?:\r?\n)+\s*([^\r\n]{2,200})"
    return re.compile(pat)


RECIPIENT_LABELS = ["RECIPIENT", "RECIPENT", "RECEPIENT", "RECIPlENT"]
RECIPIENT_BANK_LABELS = ["RECIPIENT BANK", "RECIPENT BANK", "RECEPIENT BANK"]


# -----------------------------
# Rule-based doc type + extraction
# -----------------------------
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

    # Chase wire
    if "CHASE" in u and "WIRE" in u:
        m = compile_line_after_label("SENDER").search(text)
        if m:
            fields["SenderName"] = m.group(1).strip()

        if not fields["RecipientName"]:
            for lbl in RECIPIENT_LABELS:
                m = compile_line_after_label(lbl).search(text)
                if m:
                    fields["RecipientName"] = m.group(1).strip()
                    break

        for lbl in RECIPIENT_BANK_LABELS:
            m = compile_line_after_label(lbl).search(text)
            if m:
                fields["RecipientBank"] = m.group(1).strip()
                break

        date_tail = r"\s*[:\-]?\s*([A-Za-z]+\s+\d{1,2},\s*\d{4})"
        m = compile_after_label("Wire Transfer Date", date_tail).search(text)
        if not m:
            m = compile_after_label("Today's Date", date_tail).search(text)
        if m:
            fields["TransferDate"] = m.group(1).strip()

        amount_tail = r"\s*[:\-]?\s*\$?\s*([\d]{1,3}(?:[,]\d{3})*(?:\.\d{2})?\s*[A-Z]{3})"
        m = compile_after_label("Transfer Amount", amount_tail).search(text)
        if m:
            fields["TransferAmount"] = m.group(1).strip()

        fees_tail = r"\s*[:\-]?\s*\+?\$?\s*([\d,]+(?:\.\d{2})?\s*[A-Z]{3})"
        m = compile_after_label("Transfer Fees", fees_tail).search(text)
        if not m:
            m = compile_after_label("Other Fees", fees_tail).search(text)
        if m:
            fields["Fee"] = m.group(1).strip()

    # Binance withdrawal
    if ("BINANCE" in u) or ("USDT" in u) or ("TXID" in u) or ("WITHDRAWAL" in u):
        m = compile_after_label("Amount", r"\s*[\r\n]+\s*([\d,]+\s*[A-Z]+)").search(text)
        if m:
            fields["Asset"] = m.group(1).strip()

        m = compile_after_label("Network", r"\s*[\r\n]+\s*([A-Z0-9\-]+)").search(text)
        if m:
            fields["Network"] = m.group(1).strip()

        m = compile_after_label("Address", r"\s*[\r\n]+\s*([A-Za-z0-9]+)").search(text)
        if m:
            fields["WalletAddress"] = m.group(1).strip()

        m = compile_after_label("Txid", r"\s*[\r\n]+\s*([A-Fa-f0-9]+)").search(text)
        if m:
            fields["TxID"] = m.group(1).strip()

        if not fields["TransferDate"]:
            m = re.search(r"\b(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\b", text)
            if m:
                fields["TransferDate"] = m.group(1).strip()

        if not fields["Fee"]:
            m = compile_after_label("Network fee", r"\s*[\r\n]+\s*([\d.,]+\s*[A-Z]+)").search(text)
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
    collapse_ws = st.checkbox("Collapse whitespace/newlines", value=True)

    use_ai_agent = st.checkbox("Use AI Agent (ChatGPT)", value=False)
    ai_model = st.text_input("AI model (OpenAI)", value="gpt-5")
    st.caption("Requires OPENAI_API_KEY in your environment. If missing, AI Agent will be skipped.")

    include_text_cols = st.checkbox("Include Raw/Clean text columns in CSV/Excel", value=False)

results = []

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
    for r in results:
        filename = r.get("filename")
        raw = r.get("raw_text", "") or ""
        clean = clean_ocr_text(raw, collapse_ws=collapse_ws)

        rule = extract_financial_fields_rule(clean)

        agent = {}
        if use_ai_agent:
            agent = extract_fields_with_agent(clean, model=ai_model)

        # Merge: prefer agent value, fallback to rule
        merged = {}
        for k in rule.keys():
            merged[k] = agent.get(k) or rule.get(k)

        # Conflict detection
        conflicts = []
        for k in rule.keys():
            rv = rule.get(k)
            av = agent.get(k)
            if rv and av and str(rv) != str(av):
                conflicts.append(k)

        row = {
            "Filename": filename,
            **merged,
            "AI_confidence": agent.get("confidence") if agent else None,
            "Conflicts": ";".join(conflicts) if conflicts else "",
        }

        if include_text_cols:
            row["RawText"] = raw
            row["CleanText"] = clean

        rows.append(row)

    # Excel/CSV columns: only "needed fields" by default
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
    # Ensure column order (ignore missing)
    cols = [c for c in base_cols if c in df.columns]
    df = df[cols]

    st.subheader("Preview")
    st.dataframe(df, use_container_width=True, height=560)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Files processed", len(df))
    c2.metric("DocType detected", int((df["DocumentType"] != "OTHER").sum()) if "DocumentType" in df else 0)
    c3.metric("WalletAddress detected", int(df["WalletAddress"].notna().sum()) if "WalletAddress" in df else 0)
    c4.metric("TxID detected", int(df["TxID"].notna().sum()) if "TxID" in df else 0)

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
