
# HelloRule

An AI-powered Q&A chatbot on **Indian law**, built using Retrieval-Augmented Generation (RAG).

Covers three laws — the **Constitution of India**, the **Bharatiya Nyaya Sanhita (BNS) 2023**
(criminal offences, replaced the IPC), and the **Bharatiya Nagarik Suraksha Sanhita (BNSS) 2023**
(criminal procedure, replaced the CrPC). Ask about articles, sections, schedules, amendments, or
old IPC/CrPC references — get cited answers grounded in the actual legal text.

## Tech Stack

**Backend:** FastAPI, ChromaDB, BAAI/bge-small-en-v1.5 embeddings
**Frontend:** React 19, Vite
**LLM:** Llama 3.1 8B via Groq API
**Data:** ~1,482 entries — Constitution (585), BNS (361), BNSS (536)

## Quick Start

```bash
# Backend
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key" > .env
python3 ingest.py            # loads all three laws from data/, builds the vector store
uvicorn main:app --port 8000

# Frontend
cd frontend && npm install && npm run dev
```

## How It Works

User question → Embed & search ChromaDB → Retrieve top 5 chunks → Build prompt with context → Answer via Groq/Llama 3.1 → Return with source citations.

The retriever is more than plain semantic search: it exact-matches direct number lookups
("Article 21", "Section 103 of the BNS") via metadata, and translates old-law references
(IPC 420 → BNS 318, CrPC 154 → BNSS 173) using each section's equivalence data.

## Project Structure

```
├── main.py # FastAPI server (both endpoints)
├── ingest.py # Data chunking & embedding pipeline
├── retriever.py # ChromaDB search: exact-match, IPC/CrPC translation, filtering
├── generator.py # Groq LLM integration
├── prompt.py # System prompt & prompt assembly
├── agents/ # Multi-agent layer (supervisor + per-law specialists)
│   ├── router.py # Supervisor: routes a question to the relevant law(s)
│   ├── specialist.py # Per-law expert (scoped retrieval + focused prompt)
│   ├── synthesizer.py # Merges multi-law answers
│   └── orchestrator.py # Wires router -> specialists -> synthesizer
├── evals/ # Evaluation framework
│   ├── eval_lib.py # Scoring (citation parsing, faithfulness)
│   ├── build_coverage_cases.py # Ground-truth case generator from law JSON
│   ├── eval_coverage.py # Per-law coverage eval (single pipeline)
│   ├── eval_translation.py # IPC/CrPC-to-BNS/BNSS translation eval
│   ├── eval_multiagent.py # Router + baseline-vs-multi-agent A/B
│   └── eval_multilaw_coverage.py # Dual-law coverage on cross-law questions
├── data/ # Law datasets (JSON)
└── frontend/ # React chat UI
```

## Two query pipelines

The backend exposes two ways to answer a question, sharing the same retriever and model:

| Endpoint | How it works | Best for | Cost |
|----------|--------------|----------|------|
| `POST /api/chat` | One retrieval over all laws → one LLM call | Single-law questions | 1 LLM call |
| `POST /api/chat/multiagent` | A supervisor routes to per-law specialist(s); multi-law questions run in parallel and are synthesized | Questions spanning several laws | 2–4 LLM calls |

They are complementary, not a replacement: `/api/chat` is the lean default; the multi-agent
path adds value specifically on cross-law questions (see Evaluation).

## Evaluation

### Retrieval coverage (single pipeline)

One auto-generated question per entry across all three laws (**1,481 / 1,482 cases scored**),
grading three layers independently: **retrieval** (was the correct provision fetched?),
**citation** (did the model cite the right section, and never one it didn't retrieve?), and
**faithfulness** (does the answer match the real legal text — embedding similarity, escalated
to an LLM judge only when borderline).

| Law | n | Retrieval | Citation recall | Citation precision | Faithfulness |
|-----|----|-----------|-----------------|--------------------|--------------|
| Constitution | 585 | 100% | 88.2% | 79.0% | 0.831 |
| BNS | 361 | 100% | 100% | 97.3% | 0.880 |
| BNSS | 535 | 100% | 100% | 92.0% | 0.869 |

**Retrieval is 100% across all entries** after adding exact-match metadata filtering for
direct number lookups (e.g. "Article 21") — previously as low as 21% on BNS via semantic
search alone. Zero LLM-judge escalations across the full run. The Constitution's lower
citation precision (79%) is the one open item — partly legitimate cross-referencing of
related articles rather than pure hallucination.

A separate translation eval (IPC/CrPC → BNS/BNSS) passes **26/27**.

### Baseline vs multi-agent

On a curated set, the **router** picks the correct law(s) on **14/14** questions. Comparing
the two pipelines on the same questions:

| Question type | Baseline (`/api/chat`) | Multi-agent (`/api/chat/multiagent`) |
|---------------|------------------------|--------------------------------------|
| Single-law (retrieval hit, citation precision) | 9/10, 90% | 9/10, 90% — **equivalent** |
| Multi-law (retrieval covers both laws) | 0/4 | 4/4 — **multi-agent wins** |

The takeaway: on single-law questions the two are equivalent (same retriever, so the extra
router call buys nothing). On cross-law questions the baseline's top-5 is dominated by one
law and misses the other entirely — where it appears to cite both, those citations are
ungrounded (from the model's memory, not retrieved). The multi-agent path retrieves from each
law separately and covers both. So the multi-agent value is specific to multi-law questions,
not universal.

```bash
python3 evals/build_coverage_cases.py         # generate coverage cases from the law JSON
python3 evals/eval_coverage.py --mode sample  # ~60-case retrieval sweep (--mode full for all, resumable)
python3 evals/eval_translation.py             # IPC/CrPC translation
python3 evals/eval_multiagent.py              # router accuracy + baseline-vs-multi-agent A/B
python3 evals/eval_multilaw_coverage.py       # dual-law coverage on cross-law questions
```

## Disclaimer

This application is for **educational and informational purposes only**. The constitutional data may differ from the official text, and AI-generated responses may contain inaccuracies. This is not legal advice — refer to [legislative.gov.in](https://legislative.gov.in) for the authoritative Constitution text. The creators accept no liability for decisions based on this application's responses.

## License

Open source. The Constitution of India is a public document.
