# RAG — Natural Farming (Streamlit + Gemini)

A minimal Retrieval-Augmented Generation (RAG) app for **Natural Farming** built with Streamlit.  
Upload PDFs or provide URLs, the app ingests documents, builds embeddings with `sentence-transformers`, stores them in FAISS, and generates grounded answers using Google Gemini (via `google-generativeai`).

---

## Quick start (Windows / PowerShell)

1. Clone (or create) repo and open terminal in project root:
```powershell
git clone <your-repo-url>
cd rag_nf
