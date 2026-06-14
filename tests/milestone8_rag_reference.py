# Milestone 8 Reference Version

# Contains:
# - PDF Upload
# - Text Extraction
# - Chunking
# - Embeddings
# - FAISS Search
# - Top-k Retrieval
# - Gemini Answer Generation

# Kept for debugging and learning purposes.

from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

import streamlit as st
import PyPDF2
import faiss
import numpy as np
import google.generativeai as genai

# Gemini Configuration
genai.configure(api_key="api_key")

# Gemini Model Initialization
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

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

    # User Question Input

    st.subheader("Ask Questions About The PDF")
    question = st.text_input("Enter your question")

    if question:
        # Question Embedding Generation
        question_embedding = embedding_model.encode([question])

        # Similarity Search
        distances, indices = faiss_index.search(
            np.array(question_embedding).astype("float32"),
            k=3
            )
        
        # Display Matching Chunks
        st.subheader("Top Matching Chunks")

        for rank, chunk_index in enumerate(indices[0]):
            st.write(f"Match {rank + 1}")

            st.text_area(
                f"Retrieved Chunk {rank + 1}",
                chunks[chunk_index],
                height=150
            )
        
        # Combine Retrieved Chunks
        retrieved_context = ""

        for chunk_index in indices[0]:
            retrieved_context += chunks[chunk_index]
            retrieved_context += "\n\n"

        # Prompt Creation
        prompt = f"""
        Answer the question using only the provided context.

        Context:
        {retrieved_context}

        Question:
        {question}
        """

        # Gemini Answer Generation
        response = gemini_model.generate_content(prompt)

        # Display Final Answer
        st.subheader("Generated Answer")
        st.write(response.text)
