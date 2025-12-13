# app.py
import os
import base64
import logging
from io import BytesIO

import streamlit as st
import requests
from PIL import Image
import fitz  # PyMuPDF

# --------------------------------------------------
# LOGGING
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Handwritten Answer Evaluation", layout="wide")

API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("❌ GEMINI_API_KEY not found in Streamlit secrets or environment")
    st.stop()

GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/gemini-1.5-flash:generateContent"
)

MAX_IMAGE_SIZE = (1024, 1024)

# --------------------------------------------------
# UTILS
# --------------------------------------------------
def pil_to_base64(img: Image.Image) -> str:
    img = img.convert("RGB")
    img.thumbnail(MAX_IMAGE_SIZE)

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def call_gemini_ocr_from_b64(image_b64: str) -> str:
    headers = {"Content-Type": "application/json"}

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": "Extract all handwritten text clearly."},
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": image_b64
                        }
                    }
                ]
            }
        ]
    }

    r = requests.post(
        GEMINI_ENDPOINT,
        headers=headers,
        params={"key": API_KEY},
        json=payload,
        timeout=60
    )

    # 🔴 IMPORTANT: SHOW FULL ERROR
    if r.status_code != 200:
        st.error(f"Gemini API Error: {r.status_code}")
        st.code(r.text)
        raise RuntimeError("Gemini OCR failed")

    data = r.json()

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        return ""


def extract_images_from_pdf(pdf_bytes):
    images = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=200)
        img = Image.open(BytesIO(pix.tobytes("png")))
        images.append(img)

    return images


# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("📝 AI-Powered Handwritten Answer Sheet Evaluation")
st.write("Upload a handwritten answer sheet (PDF or Image)")

uploaded_file = st.file_uploader(
    "Upload PDF / JPG / PNG",
    type=["pdf", "jpg", "jpeg", "png"]
)

if uploaded_file:
    extracted_pages = []

    with st.spinner("Processing document..."):
        try:
            if uploaded_file.type == "application/pdf":
                pdf_bytes = uploaded_file.read()
                images = extract_images_from_pdf(pdf_bytes)

                for idx, img in enumerate(images):
                    st.info(f"OCR on page {idx + 1}")
                    img_b64 = pil_to_base64(img)
                    text = call_gemini_ocr_from_b64(img_b64)
                    extracted_pages.append(text)

            else:
                img = Image.open(uploaded_file)
                img_b64 = pil_to_base64(img)
                text = call_gemini_ocr_from_b64(img_b64)
                extracted_pages.append(text)

        except Exception as e:
            st.error("❌ Error during OCR processing")
            st.exception(e)
            st.stop()

    # --------------------------------------------------
    # OUTPUT
    # --------------------------------------------------
    st.success("✅ OCR Completed Successfully")

    for i, page_text in enumerate(extracted_pages, start=1):
        st.subheader(f"📄 Page {i}")
        st.text_area(
            label=f"Extracted Text – Page {i}",
            value=page_text,
            height=300
        )
