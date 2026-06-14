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
genai.configure(api_key=api_key)

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

    with st.spinner("Processing PDF..."):

        st.success(f"Uploaded: {uploaded_file.name}")

        # PDF Text Extraction
        pdf_reader = PyPDF2.PdfReader(uploaded_file)

        extracted_text = ""

        for page in pdf_reader.pages:
            page_text = page.extract_text()

            if page_text:
                extracted_text += page_text

        # Document Chunking

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = text_splitter.split_text(extracted_text)

        # Embedding Generation

        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = embedding_model.encode(chunks)

        # FAISS Vector Store Creation

        embedding_dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(embedding_dimension)
        faiss_index.add(np.array(embeddings).astype("float32"))

        # User Question Input

        st.subheader("Ask Questions About The PDF")
        question = st.text_input("Enter your question")

        if question and question.strip():
            # Question Embedding Generation
            question_embedding = embedding_model.encode([question])

            # Similarity Search
            distances, indices = faiss_index.search(
                np.array(question_embedding).astype("float32"),
                k=3
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

            try:
                # Gemini Answer Generation
                with st.spinner("Generating answer..."):
                    response = gemini_model.generate_content(prompt)

                # Display Final Answer
                st.subheader("Generated Answer")
                st.write(response.text)
            except Exception as e:
                st.error(f"Error generating answer: {e}")