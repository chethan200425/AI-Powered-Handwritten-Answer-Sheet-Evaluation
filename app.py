# app.py
import os
import re
import uuid
import base64
import logging
from io import BytesIO

import streamlit as st
from PIL import Image
import requests
import mimetypes

# use PyMuPDF to handle PDFs (render pages to images & extract text)
import fitz  # pip install pymupdf

logging.basicConfig(level=logging.INFO)

# ----------------------
# Config & secrets
# ----------------------
# Streamlit secrets or environment variables
API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
GEMINI_ENDPOINT = st.secrets.get("GEMINI_ENDPOINT") or os.getenv(
    "GEMINI_ENDPOINT",
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
)

# Upload folder (local working copy when running locally)
UPLOAD_ROOT = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_ROOT, exist_ok=True)

# Optional preloaded sample file path (from your uploaded files)
SAMPLE_PDF = "/mnt/data/BDA MODULE 1 ,2  Assignment1.docx (2).pdf"

st.set_page_config(page_title="AI-Powered-Handwritten-Answer-Sheet-Evaluation", layout="wide")
st.title("📄 Vigilant — Auto-evaluate Handwritten Answers (Streamlit)")

st.markdown(
    """
This app extracts handwritten text/images from uploaded PDFs or images, sends to Gemini for OCR or evaluation, and shows results.
**Important:** Put your Gemini API key into Streamlit Cloud Secrets as `GEMINI_API_KEY`. See deploy instructions below.
"""
)

# ----------------------
# Helpers
# ----------------------
def render_image_from_bytes(img_bytes):
    try:
        img = Image.open(BytesIO(img_bytes))
        return img
    except Exception:
        return None

def pdf_to_images_and_text(pdf_bytes):
    """
    Use PyMuPDF (fitz) to render PDF pages to images (JPEG) and also extract page text.
    Returns list of base64 images and list of text strings (per page).
    """
    imgs_b64 = []
    texts = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            # text extraction
            page_text = page.get_text()
            texts.append(page_text)

            # render page to image (matrix 2.0 for better resolution)
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_bytes = pix.tobytes("jpeg")
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            imgs_b64.append(b64)
        doc.close()
    except Exception as e:
        logging.exception("pdf_to_images_and_text failed")
        raise
    return imgs_b64, texts

def call_gemini_ocr_from_b64(img_b64: str, mime_type="image/jpeg"):
    """
    Send a single base64 image to Gemini (as inline binary) and request extraction.
    The prompt instructs Gemini to extract the handwritten answer text only.
    """
    if not API_KEY:
        raise RuntimeError("GEMINI API key not configured. Set GEMINI_API_KEY in Streamlit secrets.")
    prompt = "Extract handwritten answer text from this image. Return only the extracted text, do not add commentary."
    request_body = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": mime_type, "data": img_b64}}
                ]
            }
        ]
    }
    resp = requests.post(f"{GEMINI_ENDPOINT}?key={API_KEY}", json=request_body, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # parse response safely
    candidates = data.get("candidates", [])
    if candidates and "content" in candidates[0]:
        parts = candidates[0]["content"].get("parts", [])
        if parts and "text" in parts[0]:
            return parts[0]["text"].strip()
    return ""

def call_gemini_evaluate(kb_text: str, student_text: str):
    """
    Ask Gemini to evaluate the student's extracted text using the question paper KB.
    Returns Gemini's textual evaluation.
    """
    if not API_KEY:
        raise RuntimeError("GEMINI API key not configured. Set GEMINI_API_KEY in Streamlit secrets.")
    prompt = f"""
You are an examiner. Use the Question Paper / Answer Key (below) to evaluate the student's handwritten answer.
Question Paper / KB:
\"\"\"{kb_text}\"\"\"

Student Answer:
\"\"\"{student_text}\"\"\"

Return an evaluation with:
- Total Marks (out of 50)
- Relevance
- Accuracy
- Missing Key Points
- Suggestions
- One-line summary feedback

Return plain text only.
"""
    request_body = {"contents": [{"parts": [{"text": prompt}]}]}
    resp = requests.post(f"{GEMINI_ENDPOINT}?key={API_KEY}", json=request_body, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    candidates = data.get("candidates", [])
    if candidates and "content" in candidates[0]:
        parts = candidates[0]["content"].get("parts", [])
        if parts and "text" in parts[0]:
            return parts[0]["text"].strip()
    return "[No evaluation returned]"

def extract_marks_from_text(evaluation_text: str):
    m = re.search(
        r"(?:Total\s*Marks|Marks\s*Awarded|Score)\s*[:=\-]?\s*([0-9]+(?:\.[0-9]+)?(?:\s*/\s*[0-9]+)?)",
        evaluation_text,
        re.I
    )
    if m:
        return m.group(1)
    return "N/A"

# ----------------------
# UI Layout - Sidebar
# ----------------------
st.sidebar.header("Upload & Settings")
use_sample = st.sidebar.checkbox("Use sample KB (uploaded file path)", value=False)
uploaded_kb = None
if use_sample:
    st.sidebar.write("Sample KB path (local):")
    st.sidebar.code(SAMPLE_PDF)
else:
    uploaded_kb = st.sidebar.file_uploader("Upload Knowledge Base (PDF or TXT)", type=["pdf", "txt"])

uploaded_answers = st.sidebar.file_uploader(
    "Upload Answer sheets (PDF or image). For multiple, use Ctrl/Cmd+Click",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if st.sidebar.button("Process uploads"):
    if not uploaded_answers and not use_sample:
        st.sidebar.error("Please upload answer file(s) or select sample KB.")
    else:
        # create unique folder
        upload_id = uuid.uuid4().hex
        upload_folder = os.path.join(UPLOAD_ROOT, upload_id)
        os.makedirs(upload_folder, exist_ok=True)

        kb_text = ""
        # handle KB: sample path or uploaded file
        if use_sample:
            # read from local path (works when testing locally)
            try:
                with open(SAMPLE_PDF, "rb") as fh:
                    pdf_bytes = fh.read()
                _, kb_pages = pdf_to_images_and_text(pdf_bytes)
                kb_text = "\n\n".join(kb_pages)
                st.success("Loaded sample KB from disk.")
            except Exception as e:
                st.error(f"Failed to open sample KB: {e}")
        elif uploaded_kb:
            fname = uploaded_kb.name
            fpath = os.path.join(upload_folder, fname)
            with open(fpath, "wb") as fh:
                fh.write(uploaded_kb.getbuffer())
            # if PDF, extract text via fitz
            if fname.lower().endswith(".pdf"):
                with open(fpath, "rb") as fh:
                    pdf_bytes = fh.read()
                _, kb_pages = pdf_to_images_and_text(pdf_bytes)
                kb_text = "\n\n".join(kb_pages)
            else:
                kb_text = uploaded_kb.getvalue().decode("utf-8")
            st.success("Knowledge base processed.")

        # process answers (images or pdfs)
        all_pages_text = []
        preview_images = []
        for f in uploaded_answers or []:
            fname = f.name
            fpath = os.path.join(upload_folder, fname)
            with open(fpath, "wb") as fh:
                fh.write(f.getbuffer())
            mime_type = f.type or mimetypes.guess_type(fname)[0]
            try:
                if fname.lower().endswith(".pdf"):
                    pdf_bytes = open(fpath, "rb").read()
                    imgs_b64, pages_text = pdf_to_images_and_text(pdf_bytes)
                    preview_images.extend(imgs_b64)
                    # OCR each image page
                    for img_b64 in imgs_b64:
                        text = call_gemini_ocr_from_b64(img_b64, mime_type="image/jpeg")
                        all_pages_text.append(text)
                elif mime_type and mime_type.startswith("image"):
                    image_bytes = open(fpath, "rb").read()
                    b64 = base64.b64encode(image_bytes).decode("utf-8")
                    preview_images.append(b64)
                    text = call_gemini_ocr_from_b64(b64, mime_type=mime_type)
                    all_pages_text.append(text)
                else:
                    st.warning(f"Unsupported file type: {fname}")
            except Exception as e:
                st.error(f"Failed processing {fname}: {e}")
        # aggregated student answer text
        student_text = "\n\n--- Page Break ---\n\n".join(all_pages_text)
        st.session_state["kb_text"] = kb_text
        st.session_state["student_text"] = student_text
        st.session_state["preview_images"] = preview_images
        st.success("All files processed. Open 'Evaluation' tab to run evaluation.")

# ----------------------
# Main tabs
# ----------------------
tab1, tab2 = st.tabs(["Preview", "Evaluation"])

with tab1:
    st.header("Preview uploaded pages")
    if "preview_images" in st.session_state and st.session_state["preview_images"]:
        for i, b64 in enumerate(st.session_state["preview_images"], 1):
            st.write(f"Page {i}")
            img = Image.open(BytesIO(base64.b64decode(b64)))
            st.image(img, use_column_width=True)
    else:
        st.info("No preview images yet. Upload files and click 'Process uploads' in the sidebar.")

with tab2:
    st.header("Evaluation")
    if "student_text" not in st.session_state:
        st.info("No extracted text found. Upload & process files first.")
    else:
        st.subheader("Extracted student text (preview)")
        st.text_area("Student text", value=st.session_state["student_text"][:10000], height=250)

        kb_trim = (st.session_state.get("kb_text") or "")[:4000]
        st.subheader("Knowledge base (first 4000 chars)")
        st.text_area("KB", value=kb_trim, height=200)

        if st.button("Run evaluation (Gemini)"):
            try:
                with st.spinner("Calling Gemini for evaluation..."):
                    evaluation = call_gemini_evaluate(kb_trim, st.session_state["student_text"])
                st.success("Evaluation complete")
                st.code(evaluation)
                marks = extract_marks_from_text(evaluation)
                st.metric("Total Marks (parsed)", marks)
                # Save on session for later
                st.session_state["evaluation"] = evaluation
                st.session_state["marks"] = marks
            except Exception as e:
                st.error(f"Evaluation failed: {e}")

# ----------------------
# Footer / notes
# ----------------------
st.markdown("---")
st.markdown(
    """
**Notes & Deployment Tips**
- Set your Gemini API key in Streamlit Cloud Secrets as `GEMINI_API_KEY`.
- The app uses PyMuPDF (`fitz`) for PDF rendering and text extraction; no Poppler needed.
- Sample local KB path (for local testing):  
  `/mnt/data/BDA MODULE 1 ,2  Assignment1.docx (2).pdf`
"""
)
