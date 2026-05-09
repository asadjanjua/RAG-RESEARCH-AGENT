# 📄 RAG Research Agent

An AI-powered document research assistant that lets you upload multiple PDFs and ask questions about them using a local RAG pipeline.

## 🚀 Features
- Upload multiple PDFs simultaneously
- Ask natural language questions about your documents
- Fully local — no API costs using LLaMA 3.2 via Ollama
- Shows source chunks used for each answer
- Chat history within session

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **Embeddings:** HuggingFace all-MiniLM-L6-v2
- **Vector DB:** ChromaDB
- **LLM:** LLaMA 3.2 via Ollama
- **Pipeline:** LangChain LCEL

## ⚙️ How to Run
1. Install Ollama from ollama.com and run `ollama pull llama3.2`
2. Clone this repo
3. Create virtual environment: `python -m venv venv`
4. Activate: `venv\Scripts\activate`
5. Install dependencies: `pip install -r requirements.txt`
6. Run: `streamlit run app.py`