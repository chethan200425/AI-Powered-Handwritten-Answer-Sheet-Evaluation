# app.py
import os
import re
import uuid
import base64
import logging
import time
from io import BytesIO
from threading import Lock

import streamlit as st
from PIL import Image
import requests
import mimetypes
import fitz  # PyMuPDF

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)

# --------------------------------------------------
# Config & Secrets
# --------------------------------------------------
API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-1.5-flash:generateContent"
)

if not API_KEY:
    st.error("❌ GEMINI_API_KEY not set in Streamlit secrets or environment")
    st.stop()

UPLOAD_ROOT = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_ROOT, exist_ok=True)

# --------------------------------------------------
# Rate Limiting (GLOBAL)
# --------------------------------------------------
RATE_LIMIT_SECONDS = 4
_last_call_time = 0
_lock = Lock()

def rate_limit():
    global _last_call_time
    with _lock:
        now = time.time()
        wait = RATE_LIMIT_SECONDS - (now - _last_call_time)
        if wait > 0:
            time.sleep(wait)
        _last_call_time = time.time()

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def pdf_to_images_and_text(pdf_bytes, max_pages=3):
    images_b64 = []
    texts = []

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for i, page in enumerate(doc):
        if i >= max_pages:
            break

        texts.append(page.get_text())

        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("jpeg")
        images_b64.append(base64.b64encode(img_bytes).decode())

    doc.close()
    return images_b64, texts

def call_gemini_ocr_from_b64(img_b64, mime_type="image/jpeg"):
    rate_limit()

    prompt = "Extract handwritten answer text only. No explanations."

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inlineData": {"mimeType": mime_type, "data": img_b64}}
            ]
        }]
    }

    r = requests.post(
        f"{GEMINI_ENDPOINT}?key={API_KEY}",
        json=payload,
        timeout=30
    )

    if r.status_code == 429:
        raise RuntimeError("Gemini OCR rate limit hit")

    r.raise_for_status()
    data = r.json()

    return (
        data.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "")
        .strip()
    )

def chunk_text(text, size=2000):
    return [text[i:i+size] for i in range(0, len(text), size)]

def call_gemini_evaluate(kb_text, student_chunk):
    rate_limit()

    prompt = f"""
You are an examiner.

QUESTION PAPER / ANSWER KEY:
\"\"\"{kb_text}\"\"\"

STUDENT ANSWER:
\"\"\"{student_chunk}\"\"\"

Evaluate and return:
- Total Marks (out of 50)
- Relevance
- Accuracy
- Missing Points
- Suggestions
- One-line feedback

Plain text only.
"""

    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    r = requests.post(
        f"{GEMINI_ENDPOINT}?key={API_KEY}",
        json=payload,
        timeout=30
    )

    if r.status_code == 429:
        raise RuntimeError("Gemini evaluation rate limit hit")

    r.raise_for_status()
    data = r.json()

    return (
        data.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "")
        .strip()
    )

def extract_marks(text):
    m = re.search(r"Total\s*Marks.*?(\d+)", text, re.I)
    return m.group(1) if m else "N/A"

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(
    page_title="AI Handwritten Answer Evaluation",
    layout="wide"
)

st.title("📄 AI-Powered Handwritten Answer Sheet Evaluation")

st.sidebar.header("Upload")

uploaded_kb = st.sidebar.file_uploader(
    "Upload Question Paper / Answer Key (PDF or TXT)",
    type=["pdf", "txt"]
)

uploaded_answers = st.sidebar.file_uploader(
    "Upload Answer Sheets (PDF or Images)",
    type=["pdf", "jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# --------------------------------------------------
# Process Uploads
# --------------------------------------------------
if st.sidebar.button("Process Files"):
    if not uploaded_kb or not uploaded_answers:
        st.error("Upload KB and at least one answer sheet")
        st.stop()

    upload_id = uuid.uuid4().hex
    folder = os.path.join(UPLOAD_ROOT, upload_id)
    os.makedirs(folder, exist_ok=True)

    # ---- KB Processing ----
    if uploaded_kb.name.lower().endswith(".pdf"):
        kb_bytes = uploaded_kb.read()
        _, kb_pages = pdf_to_images_and_text(kb_bytes, max_pages=2)
        kb_text = "\n\n".join(kb_pages)
    else:
        kb_text = uploaded_kb.read().decode()

    # ---- Answer Processing ----
    extracted_pages = []
    preview_images = []

    for f in uploaded_answers:
        data = f.read()
        mime = f.type or mimetypes.guess_type(f.name)[0]

        if f.name.lower().endswith(".pdf"):
            imgs, _ = pdf_to_images_and_text(data, max_pages=3)
            preview_images.extend(imgs)

            for img in imgs:
                extracted_pages.append(call_gemini_ocr_from_b64(img))

        else:
            b64 = base64.b64encode(data).decode()
            preview_images.append(b64)
            extracted_pages.append(call_gemini_ocr_from_b64(b64, mime))

    st.session_state["kb"] = kb_text[:4000]
    st.session_state["student"] = "\n\n".join(extracted_pages)
    st.session_state["images"] = preview_images

    st.success("✅ Files processed successfully")

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab1, tab2 = st.tabs(["Preview", "Evaluation"])

with tab1:
    st.header("Uploaded Pages")
    for i, img in enumerate(st.session_state.get("images", []), 1):
        st.image(Image.open(BytesIO(base64.b64decode(img))), caption=f"Page {i}")

with tab2:
    if "student" not in st.session_state:
        st.info("Process files first")
    else:
        st.text_area("Extracted Student Text", st.session_state["student"], height=250)

        if st.button("Run Evaluation"):
            with st.spinner("Evaluating..."):
                results = []
                chunks = chunk_text(st.session_state["student"])[:3]

                for i, ch in enumerate(chunks):
                    res = call_gemini_evaluate(st.session_state["kb"], ch)
                    results.append(f"--- Part {i+1} ---\n{res}")

                final = "\n\n".join(results)
                st.code(final)
                st.metric("Total Marks", extract_marks(final))
