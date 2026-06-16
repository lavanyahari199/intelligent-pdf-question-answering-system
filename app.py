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

    # Validate extracted text
    if not extracted_text.strip():
        raise ValueError("No readable text found in the PDF.")

    # Split document into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(extracted_text)

    # Generate embeddings for all chunks
    embedding_model = load_embedding_model()
    embeddings = embedding_model.encode(chunks)

    # Return the chunks, their corresponding embeddings, and the total number of pages in the PDF for reference
    return (
        chunks,
        embeddings,
        len(pdf_reader.pages)
    )

# Page Title
st.set_page_config(page_title="PDF Question Answering System")

# Application Main Heading
st.title("📄 Intelligent PDF Question Answering System")

# Initialize chat history in session state to keep track of the conversation flow between user questions and generated answers
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# PDF Upload
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    all_chunks = []
    all_embeddings = []
    total_pages = 0

    with st.spinner("Processing PDF..."):

        # Process the PDF and handle potential errors gracefully
        try:
            for uploaded_file in uploaded_files:
                chunks, embeddings, page_count = process_pdf(uploaded_file.getvalue())
                all_chunks.extend(chunks)
                all_embeddings.extend(embeddings)
                total_pages += page_count

        except ValueError as e:
            st.error(str(e))
            st.stop()

        chunks = all_chunks
        embeddings = np.array(all_embeddings,dtype="float32")

        # Load cached embedding model
        embedding_model = load_embedding_model()

        # Display success message with the name of the uploaded file
        st.success(f"{len(uploaded_files)} PDF(s) uploaded and processed successfully")
        
        # Display PDF statistics in a two-column layout for better visual organization
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Pages",
                total_pages
            )

        with col2:
            st.metric(
                "Chunks",
                len(chunks)
            )

        # Clear Chat Button
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

        # FAISS Vector Store Creation

        embedding_dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(embedding_dimension)
        faiss_index.add(np.array(embeddings).astype("float32"))

        # User Question Input

        st.subheader("Ask Questions About The PDF")
        question = st.chat_input("Ask a question about the uploaded PDFs")

        if question and question.strip():
                    
            # Question Embedding Generation
            question_embedding = embedding_model.encode([question])

            # Similarity Search
            distances, indices = faiss_index.search(
                np.array(question_embedding).astype("float32"),
                k=3
                )
            # Store retrieved chunk numbers with ranks
            retrieved_chunks_with_rank = [
                (rank + 1, chunk_index + 1)
                for rank, chunk_index in enumerate(indices[0])
            ]

            # Combine Retrieved Chunks
            retrieved_context = ""
            for chunk_index in indices[0]:
                retrieved_context += chunks[chunk_index]
                retrieved_context += "\n\n"

            # Display Retrieved Chunks in an Expander
            with st.expander("View Supporting Chunks"):
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

                # Gemini Answer Generation
                with st.spinner("Generating answer..."):
                    response = gemini_model.generate_content(prompt)

                # Store conversation history
                st.session_state.chat_history.append(
                    {
                        "question": question,
                        "answer": response.text,
                        "sources": retrieved_chunks_with_rank
                    }
                )

            except Exception as e:
                # Developer log
                print(f"Gemini Error: {e}")

                # User-friendly message
                st.error(
                    "Unable to generate an answer at the moment. "
                    "Please try again later."
                )

# Display complete chat conversation history
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat["question"])
    with st.chat_message("assistant"):
        st.write(chat["answer"])
        with st.expander("View Supporting Chunks"):
            for rank, chunk_number in chat["sources"]:
                st.write(f"📌 Chunk {chunk_number} (Rank: #{rank})")