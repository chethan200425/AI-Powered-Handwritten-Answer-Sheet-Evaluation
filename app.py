# app.py
import os
import re
import uuid
import base64
import logging
import time
from io import BytesIO

import streamlit as st
from PIL import Image
import requests
import mimetypes

# use PyMuPDF to handle PDFs (render pages to images & extract text)
import fitz # pip install pymupdf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------------
# Config & secrets
# ----------------------
# Streamlit secrets or environment variables
API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

# CRITICAL FIX: Ensure a correct default endpoint is used if environment variable is not set.
GEMINI_ENDPOINT = st.secrets.get("GEMINI_ENDPOINT") or os.getenv("GEMINI_ENDPOINT") or (
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
)

# Upload folder (local working copy when running locally)
UPLOAD_ROOT = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_ROOT, exist_ok=True)

# Optional preloaded sample file path (must exist on deployment machine)
SAMPLE_PDF = os.path.join(os.getcwd(), "sample_data", "assignment.pdf") 
if not os.path.exists(os.path.dirname(SAMPLE_PDF)):
    os.makedirs(os.path.dirname(SAMPLE_PDF), exist_ok=True)


st.set_page_config(page_title="AI-Powered-Handwritten-Answer-Sheet-Evaluation", layout="wide")
st.title("📄AI-Powered-Handwritten-Answer-Sheet-Evaluation")

st.markdown(
    """
This app extracts handwritten text/images from uploaded PDFs or images, sends to Gemini for OCR or evaluation, and shows results.
**Important:** Put your Gemini API key into Streamlit Cloud Secrets as `GEMINI_API_KEY`.
"""
)

# ----------------------
# Helpers
# ----------------------
def render_image_from_bytes(img_bytes):
    """Utility to open bytes as PIL Image (unused but kept for completeness)."""
    try:
        img = Image.open(BytesIO(img_bytes))
        return img
    except Exception:
        return None

@st.cache_data(show_spinner="Rendering PDF to Images and Extracting Text...")
def pdf_to_images_and_text(pdf_bytes):
    """
    Use PyMuPDF (fitz) to render PDF pages to images (JPEG) with controlled size 
    to avoid the 4MB API payload limit and also extract page text.
    Returns list of base64 images and list of text strings (per page).
    """
    imgs_b64 = []
    texts = []
    try:
        # Open document from bytes stream
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            # text extraction (optional, primarily for printed text/KB)
            page_text = page.get_text()
            texts.append(page_text)

            # --- CRITICAL FIX START ---
            # 1. Reduce resolution matrix: 2.0 is too high for large pages/files.
            # 1.5 is a good balance for handwriting OCR.
            mat = fitz.Matrix(1.5, 1.5) 
            
            # 2. Add JPEG compression control to reduce file size further.
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_bytes = pix.tobytes("jpeg", jpeg_quality=80) # Added jpeg_quality=80
            # --- CRITICAL FIX END ---
            
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            imgs_b64.append(b64)
        doc.close()
    except Exception as e:
        logging.exception("pdf_to_images_and_text failed")
        st.error(f"PDF processing failed: {e}")
        raise
    return imgs_b64, texts

@st.cache_data(show_spinner="Running OCR on handwritten page...")
def call_gemini_ocr_from_b64(img_b64: str, mime_type="image/jpeg"):
    """
    Send a single base64 image to Gemini and request handwritten text extraction.
    """
    if not API_KEY:
        raise RuntimeError("GEMINI API key not configured. Set GEMINI_API_KEY in Streamlit secrets.")
    
    prompt = "Extract handwritten answer text from this image. Return only the extracted text, do not add commentary or formatting like Markdown."
    
    request_body = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": mime_type, "data": img_b64}}
                ]
            }
        ],
        "config": {
            "temperature": 0.0 # Set low temperature for reliable extraction
        }
    }
    
    # Use f-string for endpoint + key query parameter
    resp = requests.post(f"{GEMINI_ENDPOINT}?key={API_KEY}", json=request_body, timeout=60)
    # Check for non-200 responses and raise an exception with the status code
    resp.raise_for_status() 
    data = resp.json()
    
    # Parse response safely
    candidates = data.get("candidates", [])
    if candidates and "content" in candidates[0]:
        parts = candidates[0]["content"].get("parts", [])
        if parts and "text" in parts[0]:
            return parts[0]["text"].strip()
            
    # Handle cases where the model might be blocked or returns nothing
    return data.get("promptFeedback", {}).get("blockReason", "[No text extracted. Check response data for block reasons.]")

@st.cache_data(show_spinner="Running Evaluation against KB...")
def call_gemini_evaluate(kb_text: str, student_text: str):
    """
    Ask Gemini to evaluate the student's extracted text using the question paper KB.
    Returns Gemini's textual evaluation.
    """
    if not API_KEY:
        raise RuntimeError("GEMINI API key not configured. Set GEMINI_API_KEY in Streamlit secrets.")
        
    prompt = f"""
You are a strict and fair examiner. Use the Question Paper / Answer Key (below) to thoroughly evaluate the student's handwritten answer. The answer should be assessed out of a maximum of 50 marks.

Question Paper / KB:
\"\"\"{kb_text}\"\"\"

Student Answer:
\"\"\"{student_text}\"\"\"

Provide your evaluation in a structured, plain text format with the following mandatory sections:
- **Total Marks:** [Your calculated score]/50 (Must be the first line)
- **Relevance:** Comment on how well the answer addresses the question.
- **Accuracy:** Point out any factual errors or misconceptions.
- **Missing Key Points:** List the important points from the KB that were omitted.
- **Suggestions:** Provide constructive feedback for improvement.
- **Summary Feedback:** A one-line summary of the performance.
"""
    request_body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "config": {
            "temperature": 0.1 # Keep low for consistent evaluation
        }
    }
    
    resp = requests.post(f"{GEMINI_ENDPOINT}?key={API_KEY}", json=request_body, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    
    candidates = data.get("candidates", [])
    if candidates and "content" in candidates[0]:
        parts = candidates[0]["content"].get("parts", [])
        if parts and "text" in parts[0]:
            return parts[0]["text"].strip()
            
    return "[No evaluation returned. Check API response status.]"

def extract_marks_from_text(evaluation_text: str):
    """
    Parses the evaluation text to find the Total Marks awarded.
    Looks for patterns like '50/50', '45.5', '20/50', 'Marks: 35', etc.
    """
    m = re.search(
        r"(?:Total\s*Marks|Marks\s*Awarded|Score)\s*[:=\-]?\s*([0-9]+(?:\.[0-9]+)?(?:\s*/\s*[0-9]+)?)",
        evaluation_text,
        re.I
    )
    if m:
        return m.group(1).strip()
    
    # Also check the very first line for the mandatory structure
    first_line = evaluation_text.split('\n')[0].strip()
    m = re.search(r"(\d+(\.\d+)?\s*/\s*50)", first_line)
    if m:
        return m.group(1).strip()
    
    return "N/A"

# ----------------------
# UI Layout - Sidebar
# ----------------------
st.sidebar.header("Upload & Settings")

if not API_KEY:
     st.sidebar.warning("⚠️ **Warning:** Please set the `GEMINI_API_KEY` in Streamlit secrets or environment variables.")

use_sample = st.sidebar.checkbox("Use sample KB from disk (if available)", value=False)
uploaded_kb = None

if use_sample:
    if os.path.exists(SAMPLE_PDF):
        st.sidebar.success(f"Sample KB available at: `{SAMPLE_PDF}`")
    else:
        st.sidebar.warning(f"Sample KB path is set but file not found: `{SAMPLE_PDF}`. Uncheck to upload a KB.")
else:
    uploaded_kb = st.sidebar.file_uploader(
        "Upload Knowledge Base (Question Paper/Answer Key PDF or TXT)", 
        type=["pdf", "txt"]
    )

uploaded_answers = st.sidebar.file_uploader(
    "Upload Answer sheets (PDF or image). For multiple, use Ctrl/Cmd+Click",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if st.sidebar.button("Process uploads"):
    if not (uploaded_answers or (use_sample and os.path.exists(SAMPLE_PDF))):
        st.sidebar.error("Please upload answer file(s) AND upload a KB or select the available sample KB.")
    elif not uploaded_answers:
        st.sidebar.error("Please upload at least one answer sheet.")
    else:
        # Clear cache for reproducibility
        pdf_to_images_and_text.clear()
        call_gemini_ocr_from_b64.clear()
        call_gemini_evaluate.clear()
        
        # create unique folder for temporary files
        upload_id = uuid.uuid4().hex
        upload_folder = os.path.join(UPLOAD_ROOT, upload_id)
        os.makedirs(upload_folder, exist_ok=True)

        kb_text = ""
        # 1. Handle KB: sample path or uploaded file
        with st.spinner("Processing Knowledge Base..."):
            try:
                if use_sample and os.path.exists(SAMPLE_PDF):
                    with open(SAMPLE_PDF, "rb") as fh:
                        pdf_bytes = fh.read()
                    _, kb_pages = pdf_to_images_and_text(pdf_bytes)
                    kb_text = "\n\n".join(kb_pages)
                    st.success("Loaded sample KB from disk.")
                elif uploaded_kb:
                    fname = uploaded_kb.name
                    fpath = os.path.join(upload_folder, fname)
                    with open(fpath, "wb") as fh:
                        fh.write(uploaded_kb.getbuffer())
                    
                    if fname.lower().endswith(".pdf"):
                        with open(fpath, "rb") as fh:
                            pdf_bytes = fh.read()
                        _, kb_pages = pdf_to_images_and_text(pdf_bytes)
                        kb_text = "\n\n".join(kb_pages)
                    else: # TXT file
                        kb_text = uploaded_kb.getvalue().decode("utf-8")
                    st.success("Knowledge base processed.")
                else:
                    st.warning("No Knowledge Base provided. Evaluation results may be poor.")
            except Exception as e:
                st.error(f"Failed to process Knowledge Base: {e}")
                logging.error(f"KB Processing Error: {e}")
                st.stop() # Stop execution on critical failure

        # 2. Process answers (images or pdfs)
        all_pages_text = []
        preview_images = []
        
        with st.spinner(f"Processing {len(uploaded_answers)} answer sheet(s) for OCR (This may take time)..."):
            for i, f in enumerate(uploaded_answers, 1):
                st.write(f"Processing file {i}/{len(uploaded_answers)}: **{f.name}**")
                
                fname = f.name
                fpath = os.path.join(upload_folder, fname)
                
                # Save uploaded file temporarily
                with open(fpath, "wb") as fh:
                    fh.write(f.getbuffer())
                    
                mime_type = f.type or mimetypes.guess_type(fname)[0]
                
                try:
                    if fname.lower().endswith(".pdf"):
                        pdf_bytes = open(fpath, "rb").read()
                        imgs_b64, pages_text = pdf_to_images_and_text(pdf_bytes)
                        
                        for page_num, img_b64 in enumerate(imgs_b64, 1):
                            st.info(f"-> Running OCR on **{fname}** (Page {page_num}/{len(imgs_b64)})...")
                            preview_images.append(img_b64)
                            # OCR each image page
                            text = call_gemini_ocr_from_b64(img_b64, mime_type="image/jpeg")
                            all_pages_text.append(text)
                            
                    elif mime_type and mime_type.startswith("image"):
                        image_bytes = open(fpath, "rb").read()
                        b64 = base64.b64encode(image_bytes).decode("utf-8")
                        preview_images.append(b64)
                        st.info(f"-> Running OCR on **{fname}**...")
                        text = call_gemini_ocr_from_b64(b64, mime_type=mime_type)
                        all_pages_text.append(text)
                        
                    else:
                        st.warning(f"Unsupported file type for processing: {fname}")
                        
                except Exception as e:
                    # Catch the API error here and provide more context
                    st.error(f"Failed processing {fname}: API Error. This is usually due to image size. Details: {e}")
                    logging.exception(f"File Processing Error: {fname}")

        # aggregated student answer text
        student_text = "\n\n--- Page Break ---\n\n".join(all_pages_text)
        
        # Save results to session state
        st.session_state["kb_text"] = kb_text
        st.session_state["student_text"] = student_text
        st.session_state["preview_images"] = preview_images
        
        st.sidebar.success("All files processed. Open 'Evaluation' tab to run evaluation.")

# ----------------------
# Main tabs
# ----------------------
tab1, tab2 = st.tabs(["Preview", "Evaluation"])

with tab1:
    st.header("Preview Uploaded Pages and Extracted Text")
    if "preview_images" in st.session_state and st.session_state["preview_images"]:
        for i, b64 in enumerate(st.session_state["preview_images"], 1):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"Page {i} (Image)")
                try:
                    img = Image.open(BytesIO(base64.b64decode(b64)))
                    st.image(img, use_column_width=True, caption=f"Original Page {i}")
                except Exception:
                    st.warning("Could not render image preview.")
            
            with col2:
                st.subheader(f"Page {i} (Extracted Text)")
                # Safely get the extracted text for the current page
                page_texts = st.session_state["student_text"].split("\n\n--- Page Break ---\n\n")
                extracted_text = page_texts[i-1] if i-1 < len(page_texts) else "[Extraction failed or returned no text]"
                st.code(extracted_text, language="text")
                
            st.markdown("---")
    else:
        st.info("No preview images yet. Upload files and click 'Process uploads' in the sidebar.")

with tab2:
    st.header("Evaluation")
    if "student_text" not in st.session_state or not st.session_state["student_text"]:
        st.info("No extracted text found. Upload & process files first.")
    else:
        kb_text = st.session_state.get("kb_text") or ""
        student_text = st.session_state["student_text"]

        st.subheader("Knowledge Base (for reference)")
        st.text_area("KB Text", value=kb_text[:4000] if kb_text else "[KB not loaded]", height=150)

        st.subheader("Aggregated Student Answer Text")
        st.text_area("Student Text (Full)", value=student_text, height=250)

        if st.button("Run Evaluation (Gemini)"):
            if not kb_text:
                 st.error("Cannot run evaluation. The Knowledge Base (KB) text is empty.")
            else:
                try:
                    # Clear previous evaluation cache to run fresh
                    call_gemini_evaluate.clear() 
                    
                    with st.spinner("Calling Gemini for detailed evaluation..."):
                        evaluation = call_gemini_evaluate(kb_text[:4000], student_text) 
                    
                    st.success("Evaluation complete")
                    st.markdown("### 🎓 Evaluation Results")
                    
                    # 1. Display Score
                    marks = extract_marks_from_text(evaluation)
                    st.metric("Total Marks Awarded", marks)
                    
                    # 2. Display Full Evaluation
                    st.code(evaluation, language="text")
                    
                    # Save on session for later
                    st.session_state["evaluation"] = evaluation
                    st.session_state["marks"] = marks
                    
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")
                    logging.exception("Evaluation Call Failed")

# ----------------------
# Footer / notes
# ----------------------
st.markdown("---")
st.markdown("Powered by the Gemini 2.5 Flash API for OCR and evaluation.")
