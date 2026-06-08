# 🧠 Document Intelligence System v2.0

An enterprise-grade GenAI application that 
automatically classifies, extracts entities, 
and enables natural language Q&A across 
multiple PDF documents.

## Features
- 📄 Multi-document PDF processing
- 🎯 AI-powered document classification
- 🔍 Structured entity extraction → JSON
- 🤖 Autonomous insights agent
- ⚠️ Risk assessment and flagging
- 💬 Cross-document RAG chat
- 📚 Source attribution per answer

## Tech Stack
- Python
- Google Gemini AI
- LangChain
- FAISS Vector Store
- Sentence Transformers
- Streamlit
- PyMuPDF

## How to Run

1. Clone the repository
2. Install dependencies:
   pip install -r requirements.txt
3. Create .env file:
   GEMINI_API_KEY=your_key_here
4. Run the app:
   streamlit run app.py

## Architecture
Upload PDF → Text Extraction → 
Classification → Entity Extraction → 
Insights Agent → RAG Pipeline → 
Chat Interface
