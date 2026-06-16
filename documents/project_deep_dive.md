# Intelligent PDF Question Answering System – Technical Deep Dive

## 1. Type of Data

* Unstructured text data extracted from PDF documents.
* Supports reports, research papers, manuals, and other text-based PDFs.

## 2. Data Format

Input:

* PDF (.pdf)

Intermediate Processing:

* Plain text
* Text chunks

Vector Representation:

* Dense numerical embeddings

## 3. Storage

Application Runtime:

* Streamlit Session State

Document Processing:

* Chunks stored in Python memory

Vector Storage:

* FAISS in-memory vector index

Persistent Storage:

* Not implemented (future enhancement)

## 4. Embedding Model

Model:

* all-MiniLM-L6-v2

Library:

* Sentence Transformers

Embedding Type:

* Dense semantic embeddings

Purpose:

* Convert text chunks and user questions into vector representations for similarity search.

## 5. Vector Database

Database:

* FAISS (Facebook AI Similarity Search)

Index Type:

* IndexFlatL2

Purpose:

* Fast nearest-neighbor retrieval of semantically relevant document chunks.

## 6. Metadata

Currently Stored:

* Chunk number
* Chunk rank

Not Stored:

* PDF filename
* Page number
* Chunk source document

Future Enhancement:

* Add file-level and page-level metadata for source traceability.

## 7. Search

Search Type:

* Semantic Search

Method:

* Question embedding generated using the same embedding model.
* Similarity search performed against FAISS vector index.

Similarity Metric:

* L2 Distance

## 8. Index

Index Type:

* FAISS IndexFlatL2

Characteristics:

* Exact nearest-neighbor search
* No vector compression
* Suitable for small and medium-sized datasets

## 9. Threshold

Current Implementation:

* No similarity threshold applied.

Behavior:

* Top matching chunks are always returned.

Future Enhancement:

* Apply similarity score filtering before passing chunks to the LLM.

## 10. Reranking

Current Implementation:

* Not implemented.

Behavior:

* FAISS retrieval order is used directly.

Future Enhancement:

* Add cross-encoder reranking for improved retrieval accuracy.

## 11. Retrieval Parameters

Top-K Retrieval:

* k = 3

Chunk Size:

* 1000 characters

Chunk Overlap:

* 200 characters

Temperature:

* Gemini default settings

Context Window Strategy:

* Concatenation of top 3 retrieved chunks

## 12. Large Language Model (LLM)

Model:

* Gemini 2.5 Flash

Provider:

* Google AI Studio

Purpose:

* Generate answers using retrieved document context.

Prompting Strategy:

* Context-constrained Retrieval-Augmented Generation (RAG)

Prompt Instruction:

"Answer the question using only the provided context."

## 13. Guardrails

Implemented:

* Context-restricted prompting
* Input validation
* PDF text validation
* Exception handling
* User-friendly error messages

Not Implemented:

* Toxicity filtering
* Hallucination detection
* Content moderation layer
* Confidence scoring

Future Enhancement:

* Add retrieval confidence scoring and response validation.

## Overall Architecture

PDF Upload
→ PDF Text Extraction
→ Chunking
→ Embedding Generation
→ FAISS Vector Index
→ Semantic Retrieval
→ Context Construction
→ Gemini 2.5 Flash
→ Answer Generation
→ Chat Interface Display

## Key Design Decisions

* Used Sentence Transformers for lightweight local embeddings.
* Used FAISS for fast vector similarity search.
* Used Gemini 2.5 Flash for low-latency answer generation.
* Implemented Retrieval-Augmented Generation (RAG) to reduce hallucinations.
* Added supporting chunk attribution for answer transparency.
* Added multi-PDF support to improve usability.
* Adopted a chat-based interface for better user experience.