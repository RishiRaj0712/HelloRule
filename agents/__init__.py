"""
agents/ — Multi-agent layer for LawBook India
==============================================
A supervisor/specialist architecture built ON TOP of the existing RAG stack
(retriever.py, ChromaDB, Groq). Nothing here replaces the current single-
pipeline app — it's a parallel path exposed via a new endpoint.

Components (built phase by phase):
  router.py       — supervisor: routes a question to the relevant law(s)   [Phase 1]
  specialist.py   — per-law expert agent (scoped retrieval + prompt)        [Phase 2]
  synthesizer.py  — merges multiple specialist answers                      [Phase 3]
  orchestrator.py — wires supervisor -> specialists -> synthesizer          [Phase 3]
"""
