import streamlit as st
import tempfile
import pdfplumber
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import os

# ---------------- CONFIG ----------------
genai.configure(api_key="YOUR_GEMINI_KEY")  # replace with your real key
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "models/gemini-2.5-flash"
CHUNK_SIZE, OVERLAP = 800, 150


# ---------------- HELPERS ----------------
def extract_text_pdf(file_path: str) -> str:
    """Try extracting text using pdfplumber, fallback to OCR with PyMuPDF."""
    try:
        with pdfplumber.open(file_path) as pdf:
            text = "".join([page.extract_text() or "" for page in pdf.pages])
        if text.strip():
            return text
    except Exception:
        pass
    return extract_text_ocr(file_path)


def extract_text_ocr(file_path: str) -> str:
    """OCR extraction using PyMuPDF."""
    try:
        pdf_document = fitz.open(file_path)
        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text += page.get_text("text")
        return text
    except Exception as e:
        st.error(f"OCR extraction failed: {e}")
        return ""


def process_and_embed(text: str):
    """Split text into chunks and embed with HuggingFace."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP)
    chunks = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = FAISS.from_texts(chunks, embedding=embeddings)
    return db


def generate_pdf_report(filename, tender_text, company_name, company_text, evaluation):
    """Generate PDF report with ReportLab."""
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(filename)

    story = []

    story.append(Paragraph("📑 Tender–Proposal Evaluation Report", styles["Title"]))
    story.append(Spacer(1, 20))

    story.append(Paragraph("<b>Tender Document Extract:</b>", styles["Heading2"]))
    story.append(Paragraph(tender_text[:800] + "...", styles["Normal"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph(f"<b>Company Proposal Extract ({company_name}):</b>", styles["Heading2"]))
    story.append(Paragraph(company_text[:800] + "...", styles["Normal"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>Evaluation Result:</b>", styles["Heading2"]))
    story.append(Paragraph(evaluation, styles["Normal"]))
    story.append(Spacer(1, 20))

    # Add summary table
    data = [
        ["Aspect", "Status"],
        ["Experience", "Checked"],
        ["License", "Checked"],
        ["Timeline", "Checked"],
        ["Financials", "Checked"]
    ]
    table = Table(data, colWidths=[200, 200])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(table)

    doc.build(story)


# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="Tender–Proposal Matcher", layout="wide")
st.title("📑 Tender–Proposal Matcher")

tender_pdf = st.file_uploader("Upload Tender Document (PDF)", type=["pdf"])
company_pdfs = st.file_uploader("Upload Company Proposals (up to 10 PDFs)", type=["pdf"], accept_multiple_files=True)

if st.button("Process & Match"):
    if not tender_pdf or not company_pdfs:
        st.error("⚠ Please upload both a Tender PDF and at least one Company PDF.")
    else:
        # --- Tender file ---
        temp_tender = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tender_pdf.seek(0)
        tender_data = tender_pdf.read()
        if not tender_data:
            st.error("⚠ Tender PDF seems empty.")
            st.stop()
        temp_tender.write(tender_data)
        temp_tender.flush()
        tender_text = extract_text_pdf(temp_tender.name)

        if not tender_text.strip():
            st.error("⚠ Could not extract text from Tender PDF.")
            st.stop()

        # --- Embedding ---
        st.write("🔍 Creating embeddings...")
        tender_db = process_and_embed(tender_text)

        # --- Loop over company proposals ---
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2, api_key="AIzaSyDo0N65nL2pb2cqfyojrk0wb0a0osNDfEU")
        template = """
        You are an expert evaluator.
        Compare the Tender requirements with the Company Proposal.
        Highlight matches, mismatches, and overall suitability.

        TENDER DOCUMENT:
        {tender}

        COMPANY PROPOSAL:
        {company}

        Give a clear evaluation and a match score (0–100).
        """
        prompt = PromptTemplate(template=template, input_variables=["tender", "company"])

        for company_pdf in company_pdfs:
            # Save company file
            temp_company = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            company_pdf.seek(0)
            company_data = company_pdf.read()
            if not company_data:
                st.warning(f"⚠ {company_pdf.name} seems empty, skipping.")
                continue
            temp_company.write(company_data)
            temp_company.flush()

            company_text = extract_text_pdf(temp_company.name)
            if not company_text.strip():
                st.warning(f"⚠ Could not extract text from {company_pdf.name}, skipping.")
                continue

            # --- Run evaluation ---
            chain_input = {"tender": tender_text, "company": company_text}
            response = llm.invoke(prompt.format(**chain_input))

            # --- Results ---
            st.subheader(f"📊 Evaluation Result for {company_pdf.name}")
            st.write(response.content)

            # --- Generate Report for each company ---
            report_path = os.path.join(tempfile.gettempdir(), f"tender_report_{company_pdf.name}.pdf")
            generate_pdf_report(report_path, tender_text, company_pdf.name, company_text, response.content)

            with open(report_path, "rb") as f:
                st.download_button(
                    f"📥 Download Report for {company_pdf.name}",
                    f,
                    file_name=f"tender_report_{company_pdf.name}.pdf",
                    mime="application/pdf"
                )
