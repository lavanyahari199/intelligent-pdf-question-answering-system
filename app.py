import streamlit as st
import PyPDF2
import faiss
import numpy as np

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

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

    # Document Chunking

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(extracted_text)

    # Display Chunk Information

    st.subheader("Document Chunks")
    st.write(f"Total Chunks Created: {len(chunks)}")

    # Display First Three Chunks

    for index, chunk in enumerate(chunks[:3]):
        st.write(f"Chunk {index + 1}")
        st.text_area(
            f"Chunk {index + 1} Content",
            chunk,
            height=150
        )

    # Embedding Generation

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(chunks)

    # Display Embedding Information

    st.subheader("Embedding Information")
    st.write(f"Total Embeddings Created: {len(embeddings)}")
    st.write(f"Embedding Dimension: {len(embeddings[0])}")

    # FAISS Vector Store Creation

    embedding_dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(embedding_dimension)
    faiss_index.add(np.array(embeddings).astype("float32"))

    # Display FAISS Information

    st.subheader("FAISS Vector Store")
    st.write(f"Total Vectors Stored: {faiss_index.ntotal}")