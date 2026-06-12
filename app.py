import streamlit as st
import PyPDF2

# Page Title
st.set_page_config(page_title="PDF Question Answering System")

# Application Main Heading
st.title("📄 Intelligent PDF Question Answering System")

# PDF Upload
uploaded_file = st.file_uploader(
    "Upload a PDF file",
    type=["pdf"]
)

if uploaded_file:

    st.success(f"Uploaded: {uploaded_file.name}")

    # PDF Text Extraction
    pdf_reader = PyPDF2.PdfReader(uploaded_file)

    extracted_text = ""

    for page in pdf_reader.pages:
        page_text = page.extract_text()

        if page_text:
            extracted_text += page_text

    # Display Extracted Content
    st.subheader("Extracted Text")

    st.text_area(
        "PDF Content",
        extracted_text,
        height=300
    )