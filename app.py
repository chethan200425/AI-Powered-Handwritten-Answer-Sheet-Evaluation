# app_streamlit.py
import os
import re
import uuid
import base64
import logging
import tempfile
from io import BytesIO

import requests
from pdf2image import convert_from_bytes
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

import streamlit as st
from werkzeug.utils import secure_filename  # convenient filename sanitizer
from dotenv import load_dotenv

# ---- Load .env (optional) ----
load_dotenv()
logging.basicConfig(level=logging.INFO)

# ---- Config: prefer Streamlit secrets, fallback to env vars ----
API_KEY = st.secrets.get("API_KEY") if "API_KEY" in st.secrets else os.getenv("API_KEY")
GEMINI_ENDPOINT = st.secrets.get("GEMINI_ENDPOINT") if "GEMINI_ENDPOINT" in st.secrets else os.getenv(
    "GEMINI_ENDPOINT",
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
)
POPPLER_PATH = st.secrets.get("POPPLER_PATH") if "POPPLER_PATH" in st.secrets else os.getenv(
    "POPPLER_PATH", None
)

# ---- App settings ----
st.set_page_config(page_title="Vigilant: Auto-evaluator", layout="wide")
st.title("Vigilant — Knowledge Base + Handwritten Answer Evaluator")

# ---- Upload area ----
st.markdown("Upload a **question paper / knowledge base** (PDF/DOCX) and one or more **student answer sheets** (PDF or images).")
kb_file = st.file_uploader("Upload Knowledge Base (PDF preferred)", type=["pdf", "txt"], key="kb")
answer_files = st.file_uploader("Upload Answer Sheets (PDF or image). You can upload multiple files.", type=["pdf", "png", "jpg", "jpeg", "tiff"], accept_multiple_files=True, key="answers")

# ---- Utility helpers ----
def call_gemini(prompt, img_b64=None, mime_type="image/jpeg", timeout=30):
    """
    Sends prompt (and optionally an inline image) to Gemini endpoint.
    Returns the text response or raises an exception.
    """
    if API_KEY is None:
        raise RuntimeError("API_KEY is not set. See instructions in the app header or README.")

    contents = []
    parts = [{"text": prompt}]
    if img_b64:
        parts.append({"inlineData": {"mimeType": mime_type, "data": img_b64}})
    contents.append({"parts": parts})
    request_body = {"contents": contents}

    resp = requests.post(f"{GEMINI_ENDPOINT}?key={API_KEY}", json=request_body, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    candidates = data.get("candidates", [])
    if candidates and "content" in candidates[0]:
        parts = candidates[0]["content"].get("parts", [])
        if parts and "text" in parts[0]:
            return parts[0]["text"].strip()
    return "[No text returned by Gemini]"


def extract_text_from_image_b64(img_b64: str, mime_type: str = "image/jpeg"):
    prompt = "Extract handwritten answer text from this image. Return only the extracted text, do not add comments."
    return call_gemini(prompt, img_b64=img_b64, mime_type=mime_type)


def extract_text_from_pdf_bytes(pdf_bytes: bytes, poppler_path: str = POPPLER_PATH):
    """
    Convert PDF bytes to images then use Gemini OCR on each page image.
    Returns list of extracted page texts and list of base64 preview images.
    """
    preview_imgs = []
    extracted_texts = []

    images = convert_from_bytes(pdf_bytes, dpi=200, poppler_path=poppler_path) if poppler_path else convert_from_bytes(pdf_bytes, dpi=200)
    for image in images:
        buf = BytesIO()
        image.save(buf, format="JPEG")
        b = base64.b64encode(buf.getvalue()).decode("utf-8")
        preview_imgs.append(b)
        text = extract_text_from_image_b64(b, mime_type="image/jpeg")
        extracted_texts.append(text)
    return extracted_texts, preview_imgs


# ---- Session state for storing intermediate values ----
if "kb_text" not in st.session_state:
    st.session_state["kb_text"] = ""
if "extracted_answers" not in st.session_state:
    st.session_state["extracted_answers"] = []
if "preview_imgs" not in st.session_state:
    st.session_state["preview_imgs"] = []
if "evaluation" not in st.session_state:
    st.session_state["evaluation"] = ""
if "marks" not in st.session_state:
    st.session_state["marks"] = "N/A"

# ---- Process uploads when user clicks ----
process_btn = st.button("Process uploads and Evaluate")

if process_btn:
    if not kb_file or not answer_files:
        st.error("Please upload both a knowledge base (KB) and at least one answer sheet.")
    else:
        # create temporary upload folder
        upload_id = uuid.uuid4().hex
        temp_dir = os.path.join(tempfile.gettempdir(), "vigilant_uploads", upload_id)
        os.makedirs(temp_dir, exist_ok=True)

        # Save KB locally and extract text if possible (PyMuPDF)
        kb_path = None
        kb_text = ""
        try:
            kb_fname = secure_filename(kb_file.name)
        except Exception:
            kb_fname = kb_file.name
        kb_path = os.path.join(temp_dir, kb_fname)
        with open(kb_path, "wb") as f:
            f.write(kb_file.getbuffer())

        if fitz and kb_path.lower().endswith(".pdf"):
            try:
                doc = fitz.open(kb_path)
                for page in doc:
                    kb_text += page.get_text()
                doc.close()
            except Exception:
                logging.exception("PyMuPDF extraction failed for KB.")
        else:
            # fallback: read first 20k bytes as text if it's txt
            if kb_path.lower().endswith(".txt"):
                try:
                    kb_text = open(kb_path, "r", encoding="utf-8").read()
                except Exception:
                    kb_text = ""
            else:
                logging.info("KB text extraction skipped (PyMuPDF not available or KB not PDF).")

        st.session_state["kb_text"] = kb_text

        # Process answer files
        all_extracted = []
        all_previews = []
        progress = st.progress(0)
        total_files = len(answer_files)
        for idx, file in enumerate(answer_files):
            filename = file.name
            safe_name = secure_filename(filename)
            saved_path = os.path.join(temp_dir, safe_name)
            with open(saved_path, "wb") as f:
                f.write(file.getbuffer())

            mime_type = file.type or ( "application/pdf" if filename.lower().endswith(".pdf") else "image/jpeg" )
            try:
                if filename.lower().endswith(".pdf") or mime_type == "application/pdf":
                    pdf_bytes = open(saved_path, "rb").read()
                    extracted_texts, previews = extract_text_from_pdf_bytes(pdf_bytes, poppler_path=POPPLER_PATH)
                    # join page texts for this PDF into one string (page breaks)
                    joined = "\n\n--- Page Break ---\n\n".join(extracted_texts)
                    all_extracted.append(joined)
                    all_previews.extend(previews)
                elif mime_type.startswith("image"):
                    with open(saved_path, "rb") as fh:
                        image_bytes = fh.read()
                    b64 = base64.b64encode(image_bytes).decode("utf-8")
                    all_previews.append(b64)
                    txt = extract_text_from_image_b64(b64, mime_type=mime_type)
                    all_extracted.append(txt)
                else:
                    st.warning(f"Unsupported file type: {filename}")
            except Exception as e:
                logging.exception("Failed processing file")
                st.error(f"Failed to process {filename}: {e}")
            progress.progress((idx+1)/total_files)

        st.session_state["extracted_answers"] = all_extracted
        st.session_state["preview_imgs"] = all_previews

        # Build evaluation prompt
        truncated_kb = st.session_state["kb_text"][:4000]  # keep prompt reasonable
        student_answer = "\n\n--- Answer Documents ---\n\n".join(st.session_state["extracted_answers"])
        prompt = f"""
You are a professional examiner evaluating handwritten student answers.
Question Paper (knowledge base):
\"\"\"{truncated_kb}\"\"\"

Student Answer(s):
\"\"\"{student_answer}\"\"\"

Please evaluate and return in this format:
- Total Marks (out of 50)
- Relevance
- Accuracy
- Missing Key Points
- Suggestions
- One-line summary feedback
Return only the evaluation text.
"""
        st.info("Calling evaluation API (Gemini). This may take a few seconds...")
        try:
            evaluation_text = call_gemini(prompt)
            st.session_state["evaluation"] = evaluation_text
        except Exception as e:
            logging.exception("Evaluation API call failed")
            st.error(f"Evaluation request failed: {e}")
            st.session_state["evaluation"] = "[Error calling evaluation API]"

        # extract marks
        m = re.search(r"(?:Total\s*Marks|Marks\s*Awarded|Score)\s*[:=\-]?\s*([0-9]+(?:\.[0-9]+)?(?:\s*/\s*[0-9]+)?)", st.session_state["evaluation"], re.I)
        if m:
            st.session_state["marks"] = m.group(1)
        else:
            st.session_state["marks"] = "N/A"

        st.success("Processing & evaluation complete.")

# ---- UI: show KB preview, images, extracted text and evaluation ----
st.header("Preview / Results")

if st.session_state.get("kb_text"):
    with st.expander("Knowledge Base (extracted text)"):
        st.text_area("KB Text (first 4000 chars)", value=st.session_state["kb_text"][:4000], height=200)

if st.session_state.get("preview_imgs"):
    st.subheader("Preview images from uploaded answers")
    cols = st.columns(3)
    for i, b64 in enumerate(st.session_state["preview_imgs"]):
        col = cols[i % 3]
        col.image(base64.b64decode(b64), use_column_width=True)

if st.session_state.get("extracted_answers"):
    st.subheader("Extracted Answer Texts")
    for i, txt in enumerate(st.session_state["extracted_answers"], 1):
        st.markdown(f"**Answer Document #{i}:**")
        st.text_area(f"Extracted text #{i}", value=txt, height=200)

if st.session_state.get("evaluation"):
    st.subheader("Evaluation from Gemini")
    st.write(st.session_state["evaluation"])
    st.markdown(f"**Extracted Marks:** {st.session_state.get('marks', 'N/A')}")

st.markdown("---")
st.caption("How it works: The app converts PDFs into page images (requires Poppler for PDF->image). Each image is base64-encoded and sent to Gemini for text extraction. The KB and extracted student answers are sent to Gemini for evaluation.")

# ---- Footer: setup instructions ----
with st.expander("Setup & Troubleshooting (click to expand)"):
    st.markdown(r"""
**1) API Key & Endpoint**
- Set `API_KEY` and `GEMINI_ENDPOINT` in **Streamlit secrets**:
  Create a file `.streamlit/secrets.toml` in your project:
