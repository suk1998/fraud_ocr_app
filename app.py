# app.py
import os, io, zipfile, platform, shutil
from pathlib import Path
import streamlit as st
import pandas as pd
from PIL import Image, ImageOps, ImageFilter
import pytesseract
import pdfplumber
from pdf2image import convert_from_bytes

# ---- Auto-detect Tesseract path (supports both Windows & Linux/Cloud) ----
def _auto_set_tesseract():
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

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PDF_EXT = {".pdf"}

# ---- Image preprocessing + OCR ----
def _preprocess(img: Image.Image) -> Image.Image:
    g = ImageOps.grayscale(img)
    g = ImageOps.autocontrast(g)
    g = g.filter(ImageFilter.SHARPEN)
    g = g.point(lambda p: 255 if p > 200 else (0 if p < 135 else p))
    return g

def ocr_pil(img: Image.Image, lang="eng+kor") -> str:
    return pytesseract.image_to_string(_preprocess(img), lang=lang, config="--oem 3 --psm 6")

def ocr_image_bytes(b: bytes) -> str:
    from PIL import Image
    import io as _io
    with Image.open(_io.BytesIO(b)) as im:
        return ocr_pil(im)

def pdf_text_or_ocr_bytes(b: bytes) -> str:
    # 1) Try extracting text directly (text-based PDF)
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
    # 2) If scanned PDF → convert to image and run OCR
    texts = []
    for page_img in convert_from_bytes(b, dpi=300):
        texts.append(ocr_pil(page_img))
    return "\n".join(texts)

def do_ocr_any(name: str, content: bytes) -> dict:
    ext = Path(name).suffix.lower()
    text = ""
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

# ---- UI ----
st.set_page_config(page_title="Fraud OCR Extractor", layout="wide")
st.title("🧠 Fraud OCR Extractor (Images + PDF + ZIP)")
st.caption("Upload images (JPG/PNG), PDFs, or ZIP folders — the server will perform OCR and let you download results as CSV or TXT.")

tab1, tab2 = st.tabs(["📁 Upload Files", "📦 Upload Folder (ZIP)"])

results = []

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

if results:
    df = pd.DataFrame(results, columns=["filename", "text"])
    st.subheader("📋 Preview of Results")
    st.dataframe(df, use_container_width=True, height=400)

    # ---- CSV download ----
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 Download CSV (ocr_results.csv)",
        data=csv_bytes,
        file_name="ocr_results.csv",
        mime="text/csv"
    )

    # ---- TXT (single combined) ----
    combined = []
    for _, row in df.iterrows():
        combined.append(f"===== {row['filename']} =====\n{row['text']}\n")
    combined_txt = "\n".join(combined).encode("utf-8-sig")
    st.download_button(
        "📥 Download TXT (combined_ocr.txt)",
        data=combined_txt,
        file_name="combined_ocr.txt",
        mime="text/plain"
    )

    # ---- TXT (per-file .txt inside ZIP) ----
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        for _, row in df.iterrows():
            # sanitize filename for txt
            base = Path(row["filename"]).stem
            safe = "".join(c for c in base if c not in '\\/:*?"<>|').strip() or "file"
            z.writestr(f"{safe}.txt", row["text"] or "")
    zip_buf.seek(0)
    st.download_button(
        "📥 Download TXT (per-file ZIP: ocr_texts.zip)",
        data=zip_buf.getvalue(),
        file_name="ocr_texts.zip",
        mime="application/zip"
    )
else:
    st.info("Please upload files or a ZIP folder from the tabs above.")
