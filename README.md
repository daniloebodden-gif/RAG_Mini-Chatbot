# RAG_Mini-Chatbot
A Python/Streamlit RAG chatbot using FAISS embeddings and the Gemini API, developed as a prototype for a future AI companion application inspired by the ARPG Zenless Zone Zero.

This project implements a small Retrieval-Augmented Generation (RAG) chatbot using Python and Streamlit. 
The system answers questions about a document by retrieving relevant text passages and providing them to a 
large language model for grounded responses.

The chatbot was developed as a prototype for a larger AI companion application inspired by the ARPG 
*Zenless Zone Zero*. The long-term goal of that application is to assist players with build planning, 
strategy notes, and knowledge retrieval using AI-assisted tools.


## Features
- Streamlit web interface for asking questions
- Retrieval-Augmented Generation (RAG) pipeline
- Document chunking and embedding generation
- FAISS vector database for similarity search
- Gemini API for answer generation
- Citation display for retrieved document passages
- Basic prompt-injection detection safeguard

## Model and API
Model: Gemini 2.5 Flash  
Source: Google Gemini API

Gemini was selected because it provides fast response generation and is easily integrated through a 
Python API. The model is used only for response generation, while document retrieval is handled locally 
through embeddings and FAISS vector search.

## Dataset
The chatbot uses a small markdown document (`data/zzz_notes.md`) containing example notes related to 
game strategy and build planning. The document is split into smaller chunks and embedded into a vector 
space for similarity-based retrieval.

  
## Development Process (Agile)
Sprint 1
- Set up Streamlit interface
- Connect Gemini API

Sprint 2
- Implement document chunking
- Create embeddings with sentence-transformers

Sprint 3
- Implement FAISS vector retrieval
- Add prompt injection detection
- Improve UI and debugging
