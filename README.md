
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
├── evals/ # RAG evaluation framework
│   ├── eval_lib.py # Scoring (citation parsing, faithfulness)
│   ├── build_coverage_cases.py # Ground-truth case generator from law JSON
│   └── eval_coverage.py # Systematic per-law coverage eval runner
├── data/ # Constitution dataset (JSON)
└── frontend/ # React chat UI
```

## Evaluation

A ground-truth coverage eval runs one auto-generated question per entry across all
three laws (**1,481 / 1,482 cases scored**) and grades three layers independently:
**retrieval** (was the correct provision fetched?), **citation** (did the model cite
the right section, and never one it didn't retrieve?), and **faithfulness** (does the
answer match the real legal text — embedding similarity, escalated to an LLM judge only
when borderline).

| Law | n | Retrieval | Citation recall | Citation precision | Faithfulness |
|-----|----|-----------|-----------------|--------------------|--------------|
| Constitution | 585 | 100% | 88.2% | 79.0% | 0.831 |
| BNS | 361 | 100% | 100% | 97.3% | 0.880 |
| BNSS | 535 | 100% | 100% | 92.0% | 0.869 |

**Retrieval is 100% across all entries** after adding exact-match metadata filtering for
direct number lookups (e.g. "Article 21") — previously as low as 21% on BNS via semantic
search alone. Zero LLM-judge escalations across the full run (no answer scored low enough
on faithfulness to need review). The Constitution's lower citation precision (79%) is the
one open item: it partly reflects legitimate cross-referencing of related articles rather
than pure hallucination, but tightening the prompt's citation discipline is the next step.

```bash
python3 evals/build_coverage_cases.py        # generate cases from the law JSON
python3 evals/eval_coverage.py --mode sample # quick ~60-case run
python3 evals/eval_coverage.py --mode full   # full ~1,480-case sweep (resumable via --resume)
```

## Disclaimer

This application is for **educational and informational purposes only**. The constitutional data may differ from the official text, and AI-generated responses may contain inaccuracies. This is not legal advice — refer to [legislative.gov.in](https://legislative.gov.in) for the authoritative Constitution text. The creators accept no liability for decisions based on this application's responses.

## License

Open source. The Constitution of India is a public document.
