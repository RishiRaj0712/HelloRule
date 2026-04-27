
# HelloRule

An AI-powered Q&A chatbot on the **Constitution of India**, built using Retrieval-Augmented Generation (RAG).

Ask questions about Articles, Schedules, Amendments, or Fundamental Rights — get cited answers grounded in actual constitutional text.

## Tech Stack

**Backend:** FastAPI, ChromaDB, BAAI/bge-small-en-v1.5 embeddings
**Frontend:** React 19, Vite
**LLM:** Llama 3.1 8B via Groq API
**Data:** 585 entries — 464 Articles, 12 Schedules, 106 Amendments, Preamble

## Quick Start

```bash
# Backend
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key" > .env
python3 ingest.py --json data/constitution_final.json
uvicorn main:app --port 8000

# Frontend
cd frontend && npm install && npm run dev
```

## How It Works

User question → Embed & search ChromaDB → Retrieve top 5 chunks → Build prompt with context → Stream answer via Groq/Llama 3.1 → Display with source citations.

## Project Structure

```
├── main.py # FastAPI server
├── ingest.py # Data chunking & embedding pipeline
├── retriever.py # ChromaDB search with smart filtering
├── generator.py # Groq LLM integration
├── prompt.py # System prompt & prompt assembly
├── test_suite.py # 26 automated tests
├── data/ # Constitution dataset (JSON)
└── frontend/ # React chat UI
```

## Disclaimer

This application is for **educational and informational purposes only**. The constitutional data may differ from the official text, and AI-generated responses may contain inaccuracies. This is not legal advice — refer to [legislative.gov.in](https://legislative.gov.in) for the authoritative Constitution text. The creators accept no liability for decisions based on this application's responses.

## License

Open source. The Constitution of India is a public document.
