# app.py 11/11/2025 A more detailed version of the columns
import os, io, zipfile, platform, shutil, re
from pathlib import Path
import streamlit as st
import pandas as pd
from PIL import Image, ImageOps, ImageFilter
import pytesseract
import pdfplumber
from pdf2image import convert_from_bytes

# -----------------------------
# Tesseract auto-detection
# -----------------------------
def _auto_set_tesseract():
    """Auto-detect Tesseract path on Windows/Linux."""
    if platform.system() == "Windows":
        for c in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                  r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
            if os.path.exists(c):
                pytesseract.pytesseract.tesseract_cmd = c
                return
    else:
        path = shutil.which("tesseract")
        if path:
            pytesseract.pytesseract.tesseract_cmd = path

_auto_set_tesseract()

# -----------------------------
# File type constants
# -----------------------------
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

def ocr_pil(img: Image.Image, lang="eng+kor") -> str:
    """Run OCR on a PIL image."""
    return pytesseract.image_to_string(_preprocess(img), lang=lang, config="--oem 3 --psm 6")

def ocr_image_bytes(b: bytes) -> str:
    """OCR for image bytes."""
    with Image.open(io.BytesIO(b)) as im:
        return ocr_pil(im)

def pdf_text_or_ocr_bytes(b: bytes) -> str:
    """Extract text from a PDF; if not text-based, OCR each page."""
    # 1) Try direct text extraction (text-based PDFs)
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
    # 2) Fallback to OCR (scanned PDFs)
    texts = []
    for page_img in convert_from_bytes(b, dpi=300):
        texts.append(ocr_pil(page_img))
    return "\n".join(texts)

def do_ocr_any(name: str, content: bytes) -> dict:
    """Dispatch OCR based on file extension."""
    ext = Path(name).suffix.lower()
    try:
        if ext in IMG_EXT:
            text = ocr_image_bytes(content)
        elif ext in PDF_EXT:
            text = pdf_text_or_ocr_bytes(content)
        else:
            return {"filename": name, "text": f"Unsupported file type: {ext}"}
    except Exception as e:
        return {"filename": name, "text": f"⚠️ OCR error: {e}"}
    return {"filename": name, "text": (text or "").strip()}

# -----------------------------
# Structured field extraction
# -----------------------------
def detect_doc_type(text: str) -> str:
    """Classify document type via simple keyword heuristics."""
    u = text.upper() if text else ""
    if "CHASE" in u and "WIRE" in u:
        return "CHASE_WIRE"
    if "BINANCE" in u or "USDT" in u or "TXID" in u or "WITHDRAWAL" in u:
        return "BINANCE_WITHDRAWAL"
    return "OTHER"

def extract_financial_fields(text: str) -> dict:
    """
    Extract structured fields commonly seen in bank wires and exchange withdrawals.
    The regex patterns are intentionally tolerant to OCR noise (spacing/line breaks).
    """
    fields = {
        "DocumentType": detect_doc_type(text),
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
    if not text:
        return fields

    # --- Chase Bank (wire) patterns ---
    if "CHASE" in text.upper():
        # Sender / Recipient blocks
        m = re.search(r"SENDER:\s*([\w\s.,&'-]+)", text, re.IGNORECASE)
        if m: fields["SenderName"] = m.group(1).strip()

        m = re.search(r"RECIPIENT:\s*([\w\s.,&'-]+)", text, re.IGNORECASE)
        if m: fields["RecipientName"] = m.group(1).strip()

        m = re.search(r"RECIPIENT\s*BANK[:\s]+([A-Z0-9\s.,'&()-]+)", text, re.IGNORECASE)
        if m: fields["RecipientBank"] = m.group(1).strip()

        # Dates
        m = re.search(r"(Wire\s*Transfer\s*Date|Today's\s*Date)[:\s]+([A-Za-z]+\s+\d{1,2},\s*\d{4})",
                      text, re.IGNORECASE)
        if m: fields["TransferDate"] = m.group(2).strip()

        # Amounts / Fees (e.g., $100,000,000.00 USD)
        m = re.search(r"Transfer\s*Amount[:\s]+\$?([\d,]+\.\d{2}\s*[A-Z]{3})", text, re.IGNORECASE)
        if m: fields["TransferAmount"] = m.group(1).strip()

        m = re.search(r"(Transfer\s*Fees|Other\s*Fees)[:\s]+\+?\$?([\d,]+\.\d{2}\s*[A-Z]{3})",
                      text, re.IGNORECASE)
        if m: fields["Fee"] = m.group(2).strip()

    # --- Binance (withdrawal) patterns ---
    if ("USDT" in text) or ("Txid" in text) or ("WITHDRAWAL" in text.upper()):
        # Asset + amount (e.g., Amount newline "12,000 USDT")
        m = re.search(r"Amount\s*[\r\n]+\s*([\d,]+\s*[A-Z]+)", text, re.IGNORECASE)
        if m: fields["Asset"] = m.group(1).strip()

        # Network
        m = re.search(r"\bNetwork\s*[\r\n]+\s*([A-Z0-9\-]+)", text, re.IGNORECASE)
        if m: fields["Network"] = m.group(1).strip()

        # Wallet address
        m = re.search(r"\bAddress\s*[\r\n]+\s*([A-Za-z0-9]+)", text, re.IGNORECASE)
        if m: fields["WalletAddress"] = m.group(1).strip()

        # TxID
        m = re.search(r"\bTxid\s*[\r\n]+\s*([A-Fa-f0-9]+)", text, re.IGNORECASE)
        if m: fields["TxID"] = m.group(1).strip()

        # Date (ISO-like at the bottom)
        if not fields["TransferDate"]:
            m = re.search(r"\b(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\b", text)
            if m: fields["TransferDate"] = m.group(1).strip()

        # Fee (e.g., "Network fee" line)
        if not fields["Fee"]:
            m = re.search(r"Network\s*fee\s*[\r\n]+\s*([\d.,]+\s*[A-Z]+)", text, re.IGNORECASE)
            if m: fields["Fee"] = m.group(1).strip()

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
st.title("🧠 Fraud OCR Extractor (Images + PDF + ZIP)")
st.caption("Upload images (JPG/PNG), PDFs, or ZIP — download results as Excel / CSV / TXT. Structured fields are auto-extracted for bank wires and exchange withdrawals.")

tab1, tab2 = st.tabs(["📁 Upload Files", "📦 Upload Folder (ZIP)"])

# Export options (sidebar)
with st.sidebar:
    st.subheader("💾 Export Options")
    base_input = st.text_input("Base file name", value="ocr_results",
                               help="Extension will be added automatically.")
    add_ts = st.checkbox("Append timestamp (YYYYMMDD_HHMMSS)", value=False)
    from datetime import datetime
    base_name = sanitize_basename(base_input)
    if add_ts:
        base_name = f"{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

results = []

# Tab 1: direct files
with tab1:
    up = st.file_uploader(
        "Select multiple image or PDF files",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "pdf"],
        accept_multiple_files=True
    )
    if up and st.button("🚀 Run OCR (Files)"):
        with st.spinner("Processing..."):
            for f in up:
                results.append(do_ocr_any(f.name, f.read()))

# Tab 2: ZIP folder
with tab2:
    zip_file = st.file_uploader("Upload a folder as a ZIP file", type=["zip"], key="zip")
    if zip_file and st.button("🚀 Run OCR (ZIP)"):
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

# -----------------------------
# Display + Downloads
# -----------------------------
if results:
    # Build structured rows with auto-extracted fields
    structured_rows = []
    for r in results:
        text = r.get("text", "") or ""
        fin = extract_financial_fields(text)
        structured_rows.append({
            "Filename": r.get("filename"),
            **fin,
            "FullText": text
        })

    df = pd.DataFrame(structured_rows,
                      columns=[
                          "Filename", "DocumentType", "SenderName", "RecipientName",
                          "RecipientBank", "TransferDate", "TransferAmount", "Fee",
                          "Asset", "WalletAddress", "TxID", "Network", "FullText"
                      ])

    st.subheader("📋 Detailed Preview of OCR Results")
    st.caption("Key fields (sender/recipient, bank/wallet, amount, dates) are auto-detected when possible.")
    st.dataframe(df, use_container_width=True, height=600)

    # Quick summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Files processed", len(df))
    c2.metric("Detected DocumentType", df["DocumentType"].ne("OTHER").sum())
    c3.metric("Detected WalletAddress", df["WalletAddress"].notna().sum())
    c4.metric("Detected TransferAmount", df["TransferAmount"].notna().sum())

    # CSV download
    st.download_button(
        "🧾 Download CSV",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{base_name}.csv",
        mime="text/csv"
    )

    # Excel download
    xlsx_buf = io.BytesIO()
    with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="ocr_results")
    st.download_button(
        "📘 Download Excel",
        data=xlsx_buf.getvalue(),
        file_name=f"{base_name}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # TXT (combined) download
    combined = []
    for _, row in df.iterrows():
        combined.append(
            f"===== {row['Filename']} =====\n"
            f"DocumentType: {row['DocumentType']}\n"
            f"SenderName: {row['SenderName']}\n"
            f"RecipientName: {row['RecipientName']}\n"
            f"RecipientBank: {row['RecipientBank']}\n"
            f"TransferDate: {row['TransferDate']}\n"
            f"TransferAmount: {row['TransferAmount']}\n"
            f"Fee: {row['Fee']}\n"
            f"Asset: {row['Asset']}\n"
            f"WalletAddress: {row['WalletAddress']}\n"
            f"TxID: {row['TxID']}\n"
            f"Network: {row['Network']}\n\n"
            f"{row['FullText']}\n"
        )
    st.download_button(
        "📄 Download TXT (combined)",
        data="\n".join(combined).encode("utf-8-sig"),
        file_name=f"{base_name}.txt",
        mime="text/plain"
    )

    # Per-file TXT ZIP
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        for _, row in df.iterrows():
            stem = Path(row["Filename"]).stem
            safe_stem = sanitize_basename(stem) or "file"
            txt = (
                f"Filename: {row['Filename']}\n"
                f"DocumentType: {row['DocumentType']}\n"
                f"SenderName: {row['SenderName']}\n"
                f"RecipientName: {row['RecipientName']}\n"
                f"RecipientBank: {row['RecipientBank']}\n"
                f"TransferDate: {row['TransferDate']}\n"
                f"TransferAmount: {row['TransferAmount']}\n"
                f"Fee: {row['Fee']}\n"
                f"Asset: {row['Asset']}\n"
                f"WalletAddress: {row['WalletAddress']}\n"
                f"TxID: {row['TxID']}\n"
                f"Network: {row['Network']}\n\n"
                f"{row['FullText']}\n"
            )
            z.writestr(f"{safe_stem}.txt", txt)
    zip_buf.seek(0)
    st.download_button(
        "🗂️ Download TXT (per-file ZIP)",
        data=zip_buf.getvalue(),
        file_name=f"{base_name}_texts.zip",
        mime="application/zip"
    )

else:
    st.info("Please upload files or a ZIP folder from the tabs above.")


# # app.py 10/11/2925
# import os, io, zipfile, platform, shutil, re
# from pathlib import Path
# import streamlit as st
# import pandas as pd
# from PIL import Image, ImageOps, ImageFilter
# import pytesseract
# import pdfplumber
# from pdf2image import convert_from_bytes

# # ---- Auto-detect Tesseract path (supports both Windows & Linux/Cloud) ----
# def _auto_set_tesseract():
#     if platform.system() == "Windows":
#         for c in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
#                   r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
#             if os.path.exists(c):
#                 pytesseract.pytesseract.tesseract_cmd = c
#                 return
#     else:
#         path = shutil.which("tesseract")
#         if path:
#             pytesseract.pytesseract.tesseract_cmd = path
# _auto_set_tesseract()

# IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# PDF_EXT = {".pdf"}

# # ---- Image preprocessing + OCR helpers ----
# def _preprocess(img: Image.Image) -> Image.Image:
#     """Preprocess image for better OCR accuracy."""
#     g = ImageOps.grayscale(img)
#     g = ImageOps.autocontrast(g)
#     g = g.filter(ImageFilter.SHARPEN)
#     g = g.point(lambda p: 255 if p > 200 else (0 if p < 135 else p))
#     return g

# def ocr_pil(img: Image.Image, lang="eng+kor") -> str:
#     """Run Tesseract OCR on a PIL image."""
#     return pytesseract.image_to_string(_preprocess(img), lang=lang, config="--oem 3 --psm 6")

# def ocr_image_bytes(b: bytes) -> str:
#     """Perform OCR on an image (in bytes)."""
#     from PIL import Image
#     import io as _io
#     with Image.open(_io.BytesIO(b)) as im:
#         return ocr_pil(im)

# def pdf_text_or_ocr_bytes(b: bytes) -> str:
#     """Try extracting text from PDF directly; if not possible, perform OCR page by page."""
#     try:
#         t_all, n = [], 0
#         with pdfplumber.open(io.BytesIO(b)) as pdf:
#             for p in pdf.pages:
#                 t = p.extract_text() or ""
#                 t_all.append(t)
#                 n += len(t)
#         # If text-based PDF (not scanned)
#         if n >= 50:
#             return "\n".join(t_all)
#     except Exception:
#         pass
#     # If scanned PDF → OCR
#     texts = []
#     for page_img in convert_from_bytes(b, dpi=300):
#         texts.append(ocr_pil(page_img))
#     return "\n".join(texts)

# def do_ocr_any(name: str, content: bytes) -> dict:
#     """Perform OCR depending on the file type (image or PDF)."""
#     ext = Path(name).suffix.lower()
#     try:
#         if ext in IMG_EXT:
#             text = ocr_image_bytes(content)
#         elif ext in PDF_EXT:
#             text = pdf_text_or_ocr_bytes(content)
#         else:
#             return {"filename": name, "text": f"Unsupported file type: {ext}"}
#     except Exception as e:
#         return {"filename": name, "text": f"⚠️ OCR error: {e}"}
#     return {"filename": name, "text": (text or "").strip()}

# # ---- Sanitize file base name ----
# def sanitize_basename(s: str) -> str:
#     """Remove illegal characters from file name and shorten if too long."""
#     s = s.strip()
#     if not s:
#         s = "ocr_results"
#     s = re.sub(r'[\\/:*?"<>|]+', "_", s)
#     return s[:120]

# # ---- UI ----
# st.set_page_config(page_title="Fraud OCR Extractor", layout="wide")
# st.title("🧠 Fraud OCR Extractor (Images + PDF + ZIP)")
# st.caption("Upload images (JPG/PNG), PDFs, or ZIP — download results as Excel / CSV / TXT.")

# tab1, tab2 = st.tabs(["📁 Upload Files", "📦 Upload Folder (ZIP)"])

# # ---- File export name options ----
# with st.sidebar:
#     st.subheader("💾 Export Options")
#     base_input = st.text_input("Base file name", value="ocr_results", help="The file extension is added automatically.")
#     add_ts = st.checkbox("Append timestamp (YYYYMMDD_HHMMSS)", value=False)
#     from datetime import datetime
#     base_name = sanitize_basename(base_input)
#     if add_ts:
#         base_name = f"{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# results = []

# # ---- Tab 1: Direct file upload ----
# with tab1:
#     up = st.file_uploader(
#         "Select multiple image or PDF files",
#         type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "pdf"],
#         accept_multiple_files=True
#     )
#     if up and st.button("🚀 Run OCR (Files)"):
#         with st.spinner("Processing..."):
#             for f in up:
#                 results.append(do_ocr_any(f.name, f.read()))

# # ---- Tab 2: ZIP upload ----
# with tab2:
#     zip_file = st.file_uploader("Upload a folder as a ZIP file", type=["zip"], key="zip")
#     if zip_file and st.button("🚀 Run OCR (ZIP)"):
#         with st.spinner("Extracting and processing..."):
#             with zipfile.ZipFile(io.BytesIO(zip_file.read())) as zf:
#                 for zi in zf.infolist():
#                     if zi.is_dir():
#                         continue
#                     name = zi.filename
#                     ext = Path(name).suffix.lower()
#                     if ext in (IMG_EXT | PDF_EXT):
#                         content = zf.read(zi)
#                         results.append(do_ocr_any(Path(name).name, content))

# # ---- Display and download results ----
# if results:
#     df = pd.DataFrame(results, columns=["filename", "text"])
#     st.subheader("📋 Preview of Results")
#     st.dataframe(df, use_container_width=True, height=400)

#     # --- CSV download ---
#     st.download_button(
#         "📥 Download CSV",
#         data=df.to_csv(index=False).encode("utf-8-sig"),
#         file_name=f"{base_name}.csv",
#         mime="text/csv"
#     )

#     # --- Excel download ---
#     xlsx_buf = io.BytesIO()
#     with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as w:
#         df.to_excel(w, index=False, sheet_name="ocr_results")
#     st.download_button(
#         "📘 Download Excel",
#         data=xlsx_buf.getvalue(),
#         file_name=f"{base_name}.xlsx",
#         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#     )

#     # --- TXT (combined) ---
#     combined = []
#     for _, row in df.iterrows():
#         combined.append(f"===== {row['filename']} =====\n{row['text']}\n")
#     st.download_button(
#         "📄 Download TXT (combined)",
#         data="\n".join(combined).encode("utf-8-sig"),
#         file_name=f"{base_name}.txt",
#         mime="text/plain"
#     )

#     # --- TXT per-file ZIP ---
#     zip_buf = io.BytesIO()
#     with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
#         for _, row in df.iterrows():
#             stem = Path(row["filename"]).stem
#             safe = sanitize_basename(stem) or "file"
#             z.writestr(f"{safe}.txt", row["text"] or "")
#     zip_buf.seek(0)
#     st.download_button(
#         "💾 Download TXT (per-file ZIP)",
#         data=zip_buf.getvalue(),
#         file_name=f"{base_name}_texts.zip",
#         mime="application/zip"
#     )
# else:
#     st.info("Please upload files or a ZIP folder from the tabs above.")

