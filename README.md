# Intelligent PDF Question Answering System

An AI-powered Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask natural language questions about their contents.

The system extracts text from uploaded PDFs, converts document chunks into vector embeddings, retrieves the most relevant information using semantic search, and generates context-aware answers using Google's Gemini model.

## Features

* Upload PDF documents through a Streamlit web interface
* Automatic PDF text extraction
* Document chunking with overlap preservation
* Semantic embeddings using Sentence Transformers
* FAISS vector database for similarity search
* Retrieval-Augmented Generation (RAG) workflow
* Context-aware answer generation using Gemini 2.5 Flash
* Error handling and input validation
* Streamlit caching for improved performance
* Reusable document processing pipeline
* Session-based question history

## Technology Stack

### Frontend

* Streamlit

### AI & NLP

* Google Gemini 2.5 Flash
* Sentence Transformers
* LangChain Text Splitters

### Vector Search

* FAISS

### PDF Processing

* PyPDF2

### Programming Language

* Python

## System Architecture

```text
PDF Upload
     │
     ▼
PDF Text Extraction
     │
     ▼
Document Chunking
     │
     ▼
Embedding Generation
     │
     ▼
FAISS Vector Storage
     │
     ▼
User Question
     │
     ▼
Question Embedding
     │
     ▼
Semantic Retrieval
     │
     ▼
Retrieved Context
     │
     ▼
Gemini 2.5 Flash
     │
     ▼
Generated Answer
```

## Workflow

1. User uploads a PDF document.
2. Text is extracted from all PDF pages.
3. The document is divided into overlapping chunks.
4. Embeddings are generated for each chunk.
5. Embeddings are stored in a FAISS vector index.
6. User submits a question.
7. The question is converted into an embedding.
8. FAISS retrieves the most relevant document chunks.
9. Retrieved chunks are provided to Gemini as context.
10. Gemini generates an answer based solely on the retrieved content.

## Installation

### Clone Repository

```bash
git clone https://github.com/lavanyahari199/intelligent-pdf-question-answering-system.git
cd intelligent-pdf-question-answering-system
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root directory.

```env
GEMINI_API_KEY=your_api_key_here
```

Generate your API key from Google AI Studio.

## Running the Application

```bash
streamlit run app.py
```

The application will launch in your browser at:

```text
http://localhost:8501
```

## Current Implementation Status

### Completed

* Foundation Setup
* Streamlit User Interface
* PDF Text Extraction
* Overlap Chunking
* LangChain Text Splitting
* Document Embeddings
* FAISS Vector Database
* Semantic Retrieval
* Gemini Integration
* RAG Pipeline
* Error Handling
* Model Caching
* PDF Processing Cache
* Session State Management
* Retrieved Chunk Visualization
* Source Attribution
* Question History

Current Development Stage: **Milestone 15**

## Repository Structure

```text
intelligent-pdf-question-answering-system/
│
├── data/
├── documents/
├── notebooks/
├── screenshots/
├── tests/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Author

**Lavanya H**

GitHub: https://github.com/lavanyahari199
