# Intelligent PDF Question Answering System

## Development Log and Technical Documentation

**Author:** Lavanya Balasubramaniam
**Repository:** intelligent-pdf-question-answering-system
**Last Updated:** June 2026

## Foundation Phase

### Objective

Set up the project environment, repository structure, dependency management, and version control.

### Tasks Completed

* Created project directory structure.
* Created folders for project organization.
* Initialized Python virtual environment.
* Installed required project dependencies.
* Generated `requirements.txt`.
* Initialized Git repository.
* Created GitHub repository.
* Added initial project files.

### Project Structure

```
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

### Technologies Introduced

* Python Virtual Environment
* pip
* Git
* GitHub

### Deliverables

* Project workspace
* Virtual environment
* Dependency management
* Git repository

## Milestone 1 – Streamlit UI and Git Configuration

### Objective

Create the initial web interface and configure source control exclusions.

### Tasks Completed

* Configured Streamlit application.
* Added application title.
* Added PDF upload component.
* Displayed upload confirmation message.
* Created `.gitignore`.

### Technologies Introduced

* Streamlit
* Git Ignore Rules

### Deliverables

* Basic Streamlit application
* PDF upload interface
* Git ignore configuration

## Milestone 2 – PDF Text Extraction

### Objective

Extract textual content from uploaded PDF documents.

### Tasks Completed

* Integrated PyPDF2.
* Read uploaded PDF files.
* Extracted text from all PDF pages.
* Combined extracted text into a single document string.

### Technologies Introduced

* PyPDF2

### Deliverables

* Complete document text extraction pipeline

## Milestone 3 – Overlap Chunking

### Objective

Split large documents into smaller overlapping sections to preserve context.

### Tasks Completed

* Implemented chunking logic.
* Configured chunk size of 1000 characters.
* Configured chunk overlap of 200 characters.
* Verified chunk generation.

### Deliverables

* Chunk generation workflow

## Milestone 4 – LangChain Text Splitter

### Objective

Replace manual chunking with a production-ready text splitting solution.

### Tasks Completed

* Integrated `RecursiveCharacterTextSplitter`.
* Automated document chunk creation.
* Preserved contextual continuity across chunks.

### Technologies Introduced

* LangChain Text Splitters

### Deliverables

* Structured document chunking pipeline

## Milestone 5 – Document Embeddings

### Objective

Convert text chunks into vector representations suitable for semantic search.

### Tasks Completed

* Integrated Sentence Transformers.
* Loaded `all-MiniLM-L6-v2` embedding model.
* Generated embeddings for all chunks.

### Technologies Introduced

* Sentence Transformers
* Embeddings

### Deliverables

* Vector representation of document chunks

## Milestone 6 – FAISS Vector Database

### Objective

Store embeddings efficiently for similarity search.

### Tasks Completed

* Integrated FAISS.
* Created vector index.
* Stored document embeddings in FAISS.

### Technologies Introduced

* FAISS

### Deliverables

* Vector database implementation

## Milestone 7 – Semantic Retrieval Using FAISS

### Objective

Retrieve the most relevant document chunks for a user question.

### Tasks Completed

* Embedded user questions.
* Performed nearest-neighbor similarity search.
* Retrieved top matching chunks.
* Displayed retrieved content for validation.

### Technologies Introduced

* Semantic Search
* Vector Similarity Retrieval

### Deliverables

* Retrieval layer of the RAG pipeline

## Milestone 8 – RAG Pipeline with Gemini Integration

### Objective

Generate contextual answers using retrieved document content.

### Tasks Completed

* Created Google AI Studio account.
* Generated Gemini API key.
* Configured Gemini API.
* Constructed prompts using retrieved context.
* Integrated Gemini 2.5 Flash.
* Generated answers from document context.

### Technologies Introduced

* Google Gemini API
* Retrieval Augmented Generation (RAG)

### Additional Configuration

* Created `.env` file.
* Added `.env` to `.gitignore`.
* Created `tests/test_gemini_connection.py`.
* Created Milestone 8 reference copy for debugging.

### Deliverables

* End-to-end RAG question answering system

## Milestone 9 – UI Cleanup and Error Handling

### Objective

Improve application usability and runtime stability.

### Tasks Completed

* Removed debugging outputs.
* Added PDF processing spinner.
* Added answer generation spinner.
* Added input validation.
* Added Gemini exception handling.

### Deliverables

* Cleaner user interface
* Improved user feedback
* Better runtime stability

## Milestone 10 – Model Caching for Performance Optimization

### Objective

Reduce application startup latency by reusing loaded AI models.

### Tasks Completed

* Implemented `@st.cache_resource`.
* Cached Sentence Transformer model.
* Cached Gemini model.

### Performance Impact

* Models are loaded once and reused across reruns.
* Reduced repeated initialization overhead.

### Deliverables

* Faster application response times

## Milestone 11 – PDF Processing Cache

### Objective

Avoid repeated document processing when users ask multiple questions on the same PDF.

### Tasks Completed

* Implemented `@st.cache_data`.
* Created reusable PDF processing function.
* Cached text extraction.
* Cached chunk generation.
* Cached embedding generation.
* Reused processed document data across reruns.

### Technologies Introduced

* Streamlit Data Caching

### Performance Impact

* Previously uploaded PDFs are processed only once.
* Reduced computational overhead.
* Faster repeated queries.

### Deliverables

* Optimized document processing workflow

## Milestone 12 – Session State Answer Management

### Objective

Improve user experience by managing answer display across Streamlit reruns and preventing stale answers from remaining visible while new answers are being generated.

### Tasks Completed

* Initialized Streamlit Session State.
* Created persistent answer storage using `st.session_state`.
* Added answer placeholder using `st.empty()`.
* Cleared previous answers before generating new responses.
* Stored Gemini-generated answers in Session State.
* Displayed answers using a dynamic placeholder.

### Technologies Introduced

* Streamlit Session State
* Streamlit Placeholder Components

### User Experience Improvement

* Eliminated stale answer display during new question processing.
* Improved answer lifecycle management.
* Provided cleaner interaction flow between consecutive questions.

### Deliverables

* Session-based answer management.
* Dynamic answer rendering.
* Enhanced user experience during repeated queries.

## Milestone 13 – Retrieved Chunks Display

### Objective

Increase transparency of the RAG pipeline by displaying the document chunks retrieved during semantic search.

### Tasks Completed

* Added retrieved chunk visualization.
* Integrated Streamlit expander component.
* Displayed top matching chunks returned by FAISS.
* Organized retrieved context into ranked sections.

### Technologies Introduced

* Streamlit Expander Component

### User Experience Improvement

* Improved transparency of retrieval process.
* Allowed users to inspect source context.
* Demonstrated semantic search functionality.

### Deliverables

* Retrieved chunk visualization.
* Improved RAG explainability.
* Enhanced application transparency.

## Milestone 14 – Source Attribution

### Objective

Improve answer transparency by displaying the document chunks used during answer generation.

### Tasks Completed

* Stored retrieved chunk identifiers from FAISS search results.
* Added source attribution section below generated answers.
* Displayed retrieved chunk references used for answer generation.
* Linked generated responses to supporting document context.

### Technologies Introduced

* Retrieval Source Attribution

### User Experience Improvement

* Increased answer transparency.
* Improved trust in generated responses.
* Enabled verification of supporting context.
* Made retrieval results more explainable.

### Deliverables

* Source attribution system.
* Improved answer explainability.
* Enhanced RAG transparency.

# Milestone 15 – Question History

### Objective

Maintain a history of user questions asked during the current application session for improved usability and traceability.

### Tasks Completed

* Added question history tracking using Streamlit Session State.
* Initialized question history storage.
* Stored user questions dynamically during runtime.
* Prevented duplicate question entries.
* Displayed question history below generated answers.
* Ordered questions from newest to oldest.

### Technologies Introduced

* Streamlit Session State

### Benefits

* Allows users to review previously asked questions.
* Improves interaction tracking within a session.
* Enhances overall user experience.

### Deliverables

* Session-based question history tracking.
* Dynamic question history display.
* Duplicate question prevention.

# Milestone 16 – Multi-PDF Support and Chat Interface

### Objective

Transform the application from a single-document question answering system into a conversational multi-document RAG application with an improved user experience.

### Tasks Completed

* Added support for uploading multiple PDF documents simultaneously.
* Processed and combined document chunks from multiple PDFs.
* Aggregated embeddings from all uploaded documents into a unified FAISS index.
* Added PDF statistics including total page count and chunk count.
* Replaced the traditional question-answer display with a chat-style interface.
* Introduced session-based conversation history using Streamlit Session State.
* Stored questions, answers, and supporting chunk references within chat history.
* Added supporting chunk attribution for every generated answer.
* Added Clear Chat functionality to reset conversation history.
* Simplified session state management by removing legacy answer and question-history tracking.
* Improved overall application usability and interaction flow.

### Technologies Introduced

* Streamlit Chat Components
* Multi-Document Retrieval
* Session-Based Conversation Management

### User Experience Improvement

* Users can query multiple documents simultaneously.
* Previous conversations remain visible throughout the session.
* Supporting chunks are displayed alongside each answer.
* Chat interactions provide a more natural conversational experience.
* Users can clear conversations without re-uploading documents.

### Deliverables

* Multi-PDF document support.
* Chat-based user interface.
* Persistent conversation history.
* Supporting chunk attribution per response.
* Improved RAG application usability.

## Supporting Assets Created During Development

### Environment Configuration

* `.env`
* `.gitignore`

### Testing Utilities

* `tests/test_gemini_connection.py`
* `tests/milestone8_rag_reference.py`

### Repository Documentation

* `README.md`
* `requirements.txt`
