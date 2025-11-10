# app.py
import os, io, zipfile, tempfile, platform, shutil, re
from pathlib import Path
import streamlit as st
import pandas as pd
from PIL import Image, ImageOps, ImageFilter
import pytesseract
import pdfplumber
from pdf2image import convert_from_bytes

# ---- Tesseract 자동 경로 (Cloud는 리눅스, 로컬은 윈도우 모두 지원) ----
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

IMG_EXT = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
PDF_EXT = {".pdf"}

# ---- 전처리 + OCR ----
def _preprocess(img: Image.Image) -> Image.Image:
    g = ImageOps.grayscale(img)
    g = ImageOps.autocontrast(g)
    g = g.filter(ImageFilter.SHARPEN)
    g = g.point(lambda p: 255 if p > 200 else (0 if p < 135 else p))
    return g

def ocr_pil(img: Image.Image, lang="eng") -> str:
    return pytesseract.image_to_string(_preprocess(img), lang=lang, config="--oem 3 --psm 6")

def ocr_image_bytes(b: bytes) -> str:
    with Image.open(io.BytesIO(b)) as im:
        return ocr_pil(im)

def pdf_text_or_ocr_bytes(b: bytes) -> str:
    # 1) 텍스트 PDF 시도
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
    # 2) 스캔 PDF → 이미지화 후 OCR
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
            return {"filename": name, "chars": 0, "text": f"Unsupported: {ext}"}
    except Exception as e:
        return {"filename": name, "chars": 0, "text": f"⚠️ OCR error: {e}"}
    return {"filename": name, "chars": len(text or ""), "text": (text or "").strip()}

# ---- UI ----
st.set_page_config(page_title="Fraud OCR Extractor", layout="wide")
st.title("🧠 Fraud OCR Extractor (Images + PDF + ZIP)")
st.caption("이미지(JPG/PNG)와 PDF, 또는 ZIP(폴더)을 업로드하면 서버에서 OCR 후 엑셀로 내려줍니다.")

tab1, tab2 = st.tabs(["📁 개별 파일 업로드", "📦 폴더(ZIP) 업로드"])

results = []

with tab1:
    up = st.file_uploader(
        "이미지/PDF 여러 개 선택", type=["jpg","jpeg","png","bmp","tif","tiff","pdf"],
        accept_multiple_files=True
    )
    if up and st.button("🚀 OCR 실행 (개별 파일)"):
        with st.spinner("처리 중..."):
            for f in up:
                results.append(do_ocr_any(f.name, f.read()))

with tab2:
    zip_file = st.file_uploader("폴더를 ZIP으로 업로드", type=["zip"], key="zip")
    if zip_file and st.button("🚀 OCR 실행 (ZIP)"):
        with st.spinner("압축 해제 및 처리 중..."):
            with zipfile.ZipFile(io.BytesIO(zip_file.read())) as zf:
                for zi in zf.infolist():
                    if zi.is_dir(): 
                        continue
                    name = zi.filename
                    ext = Path(name).suffix.lower()
                    if ext in IMG_EXT | PDF_EXT:
                        content = zf.read(zi)
                        results.append(do_ocr_any(Path(name).name, content))

if results:
    df = pd.DataFrame(results)
    st.subheader("📋 결과 미리보기")
    st.dataframe(df, use_container_width=True, height=400)

    # 엑셀 파일 생성
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="ocr_results")
    st.download_button(
        "📥 엑셀 다운로드 (ocr_results.xlsx)",
        data=out.getvalue(),
        file_name="ocr_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("좌측 탭에서 파일 또는 ZIP을 업로드하세요.")
