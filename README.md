# Multi-PDF Conversational AI Assistant using Gemini, LangChain, FAISS, and Gradio

This project is an intelligent document question-answering system that enables users to upload and analyze multiple PDF files simultaneously. The application extracts textual information from uploaded PDFs, processes the content into manageable chunks, converts them into vector embeddings using Google Generative AI Embeddings, and stores them in a FAISS vector database for efficient semantic retrieval.

When a user submits a query, the system performs semantic similarity search across all uploaded documents to identify the most relevant content. The retrieved information is then passed to Google's Gemini Large Language Model, which generates accurate, context-aware, and human-like responses based exclusively on the uploaded documents.

The project leverages Retrieval-Augmented Generation (RAG) architecture to minimize hallucinations and improve response accuracy by grounding answers in the actual content of the uploaded PDFs.

The user-friendly Gradio interface provides a seamless experience for uploading multiple documents, asking questions, and receiving instant AI-generated insights.

# Problem Statement

Organizations, students, researchers, and professionals often need to extract information from multiple lengthy PDF documents. Manually searching through hundreds of pages is time-consuming and inefficient.

Traditional search methods rely on keyword matching, which often fails to understand the semantic meaning behind user queries. Furthermore, large language models may generate inaccurate responses when they lack access to domain-specific documents.

There is a need for an intelligent system capable of:

Processing multiple PDF documents simultaneously.
Understanding natural language queries.
Retrieving relevant information semantically.
Generating accurate answers grounded in document content.
Providing a user-friendly conversational interface.

# Proposed Solution

The proposed solution is a Retrieval-Augmented Generation (RAG) based AI assistant that combines document retrieval and generative AI capabilities.

# Workflow
User uploads one or more PDF documents.
Text is extracted using PyPDF2.
Extracted text is divided into smaller chunks.
Google Embeddings convert text chunks into numerical vectors.
FAISS stores and indexes vectors for fast similarity search.
User asks a question.
Relevant document chunks are retrieved through semantic search.
Retrieved context is sent to Gemini Pro/Flash model.
Gemini generates a context-aware response.
The answer is displayed through the Gradio interface.
Technologies Used
Technology	Purpose
Gradio	Web Interface
Google Generative AI	LLM & Embeddings
LangChain	RAG Pipeline Management
PyPDF2	PDF Text Extraction
FAISS	Vector Database
Python Dotenv	Environment Variable Management
LangChain Google GenAI	Gemini Integration

              
# Model Used
Google Gemini 2.5 Flash

Gemini 2.5 Flash serves as the primary Large Language Model responsible for understanding user queries, processing retrieved document context, and generating accurate responses. The model offers:

Fast inference speed
Strong reasoning capabilities
Context-aware answer generation
Efficient integration with LangChain
Cost-effective deployment
Embedding Model

Google Generative AI Embeddings are used to convert text chunks into dense vector representations that capture semantic meaning, enabling highly relevant document retrieval.

# Key Features
Multi-PDF Upload Support
Semantic Search Capability
Retrieval-Augmented Generation (RAG)
Google Gemini Integration
Fast Vector Search using FAISS
Context-Aware Responses
User-Friendly Gradio Interface
Reduced Hallucination through Grounded Responses

# System Architecture
<img width="1536" height="1024" alt="ChatGPT Image May 30, 2026, 11_17_45 PM" src="https://github.com/user-attachments/assets/94e07a67-0f3b-462e-8d6f-16a078571767" />

# Data Flow Diagram
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/cfa9032f-0b42-45b3-8086-79887f4ecb41" />

