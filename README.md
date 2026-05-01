🌐 **Live Demo:** [Click here to open app](https://jrkhmbcjyig6na5nqfebwb.streamlit.app/)

# AI Research Assistant — RAG System

A fully functional Retrieval-Augmented Generation (RAG) based AI system that allows users to query research documents and receive accurate, citation-backed answers.

Built entirely using free and open-source tools, this project requires no paid APIs and can run locally with minimal setup.

---

## Overview

This system enables users to interact with research papers and documents using natural language.

Instead of relying on LLM memory, it retrieves relevant document sections and uses them as grounded context to generate responses. This ensures accuracy, traceability, and reduced hallucination.

---

## Features

### Document Understanding
- Reads and processes PDF documents  
- Extracts structured content from raw text  
- Supports any domain or topic  

---

### Question Answering
- Answers questions using document context only  
- Provides precise file name and page number citations  
- Handles multi-document querying  

---

### Conversation Memory
- Maintains chat history  
- Supports follow-up questions with context awareness  

---

### Document Analysis
Automatically extracts:
- Abstract  
- Summary  
- Methods  
- Conclusion  
- Key findings  

---

### Upload Mode
- Upload any PDF at runtime  
- Perform instant analysis and Q&A  
- Works independently from the main knowledge base  

---

### Anti-Hallucination System
- Strict prompt constraints to avoid fabricated answers  
- Faithfulness evaluation integrated  
- Fallback response when answer is not found  
- Filters out reference and bibliography sections  

---

### User Interface
- Clean and interactive UI built with Streamlit  
- Chat-based interaction  
- Source viewer and sidebar insights  

---

## rchitecture (RAG Pipeline)

The system follows a standard Retrieval-Augmented Generation pipeline:

- User submits a question

- Question is converted into a vector embedding

- FAISS searches for the most relevant text chunks

- Irrelevant sections (e.g., references) are filtered out

- A structured prompt is created using retrieved context

- The prompt is sent to the LLM (LLaMA 3.3 via Groq)

- The model generates a grounded answer with citations

- The response is stored in chat memory


---

## Configuration

All configurable parameters are centralized in `config.py`.

You can adjust:

- Model selection for LLM inference  
- Embedding model  
- Chunk size and overlap  
- Number of retrieved results  
- Reference page filtering behavior  

No changes to core logic files are required.

---

## Evaluation

The system was tested using an **LLM-as-judge evaluation method** after each development phase.

### Results:

- **Faithfulness:** No hallucinations observed in tested scenarios  
- **Citation Accuracy:** Correct file names and page numbers  
- **Retrieval Quality:** Relevant chunks consistently returned  
- **Multi-document Handling:** Works across multiple PDFs  
- **Memory Handling:** Supports follow-up queries using recent context  

---

## Tech Stack

- Python 3.10+  
- PyMuPDF for PDF parsing  
- sentence-transformers for embeddings  
- FAISS for vector search  
- Groq API for LLaMA 3.3 70B inference  
- Streamlit for UI  
- Python-dotenv for environment management  
- ChromaDB (optional) as alternative vector store  
