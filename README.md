# Local RAG System

A Streamlit app for asking questions about PDF files, web pages, and MySQL data. It creates embeddings with Ollama, stores them in a persistent local Chroma collection, and uses a local Ollama chat model to answer questions with retrieved context.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) running locally
- MySQL only if you plan to index a database

## Setup

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull llama3.2:3b
streamlit run app.py
```

Start Ollama before launching the app (`ollama serve` if it is not already running). Open the local address Streamlit prints in the terminal.

## Use

1. Choose a PDF, enter a web page URL, or supply MySQL connection details.
2. Click the corresponding **Index** button. You can add multiple sources.
3. Ask a question in the chat panel. Expand **Retrieved sources** to see the supporting chunks and cosine distances (lower is closer).
4. **Clear index** removes all indexed content and clears the current chat.

The vector index is stored in `chroma_db/` and is not committed to Git. PDF files are processed in memory, not saved to the repository. MySQL indexing currently reads **every table and row**; use a dedicated read-only user and a database you are permitted to index. Large databases may take a long time or consume substantial memory. Web pages requiring JavaScript rendering will not be extracted correctly. Image-only PDFs require OCR, which is not included.

This app is intended for local use. Do not expose Streamlit to untrusted users: URL fetching does not block internal network addresses, and the database input can read accessible tables. The model can still produce incorrect answers; check the retrieved excerpts before relying on an answer.

## Models

The default models are `nomic-embed-text` for embeddings and `llama3.2:3b` for chat. Change `EMBED_MODEL` and `CHAT_MODEL` at the top of `app.py` if you use different Ollama models. Changing the embedding model requires clearing and rebuilding the index.
