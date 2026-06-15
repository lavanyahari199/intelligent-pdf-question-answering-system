from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

import streamlit as st
import PyPDF2
import faiss
import numpy as np
import google.generativeai as genai
import io

# Caching Models for Performance Optimization
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")
@st.cache_resource
def load_gemini_model():
    return genai.GenerativeModel("gemini-2.5-flash")

# Gemini Configuration
genai.configure(api_key=api_key)

# Gemini Model Initialization
gemini_model = load_gemini_model()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Caching PDF Processing for Performance Optimization
@st.cache_data
def process_pdf(pdf_bytes):

    # Convert uploaded PDF bytes into a readable PDF object    
    pdf_reader = PyPDF2.PdfReader(
        io.BytesIO(pdf_bytes)
    )
    extracted_text = ""

    # Extract text from all pages
    for page in pdf_reader.pages:
        page_text = page.extract_text()

        if page_text:
            extracted_text += page_text

    # Split document into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(extracted_text)

    # Generate embeddings for all chunks
    embedding_model = load_embedding_model()
    embeddings = embedding_model.encode(chunks)

    # Return processed data and cache it
    return chunks, embeddings

# Page Title
st.set_page_config(page_title="PDF Question Answering System")

# Application Main Heading
st.title("📄 Intelligent PDF Question Answering System")

# Session State Initialization
if "answer" not in st.session_state:
    st.session_state.answer = ""

# PDF Upload
uploaded_file = st.file_uploader(
    "Upload a PDF file",
    type=["pdf"]
)

if uploaded_file:

    with st.spinner("Processing PDF..."):

        # Process PDF only once and reuse cached results
        chunks, embeddings = process_pdf(uploaded_file.getvalue())

        # Load cached embedding model
        embedding_model = load_embedding_model()

        st.success(f"Uploaded: {uploaded_file.name}")

        # FAISS Vector Store Creation

        embedding_dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(embedding_dimension)
        faiss_index.add(np.array(embeddings).astype("float32"))

        # User Question Input

        st.subheader("Ask Questions About The PDF")
        question = st.text_input("Enter your question")

        # Placeholder for answer display
        answer_placeholder = st.empty()

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

            # Display Retrieved Chunks in an Expander
            with st.expander("View Retrieved Chunks"):
                for rank, chunk_index in enumerate(indices[0]):
                    st.markdown(f"### Chunk {rank + 1}")
                    st.text_area(
                        f"Retrieved Chunk {rank + 1}",
                        chunks[chunk_index],
                        height=150
                    )

            # Prompt Creation
            prompt = f"""
            Answer the question using only the provided context.

            Context:
            {retrieved_context}

            Question:
            {question}
            """

            try:
                # Clear previous answer from UI
                answer_placeholder.empty()
                st.session_state.answer = ""

                # Gemini Answer Generation
                with st.spinner("Generating answer..."):
                    response = gemini_model.generate_content(prompt)

                # Store the generated answer in session state
                st.session_state.answer = response.text

                # Display Final Answer
                if st.session_state.answer:
                    answer_placeholder.subheader("Generated Answer")
                    answer_placeholder.write(st.session_state.answer)
                
            except Exception as e:
                st.error(f"Error generating answer: {e}")