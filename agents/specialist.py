"""
specialist.py — Per-law Specialist agent  [Phase 2]
=====================================================
A LawAgent is an expert in ONE law. It's deliberately just the existing RAG
pipeline, narrowed:

    retrieve (scoped to its law)  ->  build a law-focused prompt  ->  generate

Key ideas demonstrated here:
  • Scoping — retrieval is forced to a single law via retriever.search_scoped().
  • Dependency injection — the Retriever and Generator are passed IN and SHARED
    across all specialists, so the ~130MB embedding model loads once, not once
    per agent.
  • Parameterize, don't duplicate — one class, instantiated per law, instead of
    three near-identical pipelines.

Reuses the existing repo's building blocks (retriever.py, generator.py, and
prompt.py's chunk-formatting/source-extraction helpers).
"""

import sys
from pathlib import Path

# agents/ is a package next to the repo-root modules (retriever.py, etc.).
# When you run a script inside agents/, Python puts agents/ on the path, not
# the repo root — so add the repo root so `import retriever` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prompt import format_chunk_for_context, extract_sources_from_chunks  # noqa: E402

# Each specialist gets a focused system prompt. Contrast with prompt.py's
# generalist SYSTEM_PROMPT, which must describe all three laws and hedge
# between them — a specialist can be single-minded, which improves answer
# quality on its own turf.
SPECIALIST_PROMPTS = {
    "Constitution": """You are a Constitution of India specialist. You answer ONLY from the \
provided constitutional text (Articles, Parts, Schedules, Amendments, Preamble).

Rules:
- Use ONLY the provided context. If it doesn't contain the answer, say so plainly.
- Cite every claim: "According to Article 21 (Part III — Fundamental Rights)...".
- If a provision is repealed/omitted/abrogated (e.g. Article 370), state that first.
- Write in plain English for a non-lawyer. End with "Sources used: ...".""",

    "BNS": """You are a Bharatiya Nyaya Sanhita (BNS), 2023 specialist — India's substantive \
criminal law (offences and punishments), which replaced the IPC.

Rules:
- Use ONLY the provided context. If it doesn't contain the answer, say so plainly.
- Cite every claim by BNS section: "Under Section 103 of the BNS...".
- If the user names an IPC section, give the BNS equivalent from the context (the chunks \
carry "[Old IPC: N]" tags).
- Write in plain English for a non-lawyer. End with "Sources used: ...".""",

    "BNSS": """You are a Bharatiya Nagarik Suraksha Sanhita (BNSS), 2023 specialist — India's \
criminal PROCEDURE law (arrest, bail, FIR, investigation, trial, appeals), which replaced the CrPC.

Rules:
- Use ONLY the provided context. If it doesn't contain the answer, say so plainly.
- Cite every claim by BNSS section: "Under Section 173 of the BNSS...".
- If the user names a CrPC section, give the BNSS equivalent from the context (the chunks \
carry "[Old CrPC: N]" tags).
- Write in plain English for a non-lawyer. End with "Sources used: ...".""",
}

# The separator convention that generator.py's _split_prompt() uses to divide
# the assembled prompt into system + user messages. We reuse it so we can reuse
# the Generator (and its rate-limit retry logic) unchanged.
_SEP = "=" * 60


class LawAgent:
    """A specialist for one law. Shares the retriever/generator passed in."""

    def __init__(self, law: str, retriever, generator):
        if law not in SPECIALIST_PROMPTS:
            raise ValueError(f"Unknown law '{law}' — expected one of {list(SPECIALIST_PROMPTS)}")
        self.law = law
        self.retriever = retriever
        self.generator = generator
        self.system_prompt = SPECIALIST_PROMPTS[law]

    def _build_prompt(self, question: str, chunks: list[dict]) -> str:
        """system prompt + retrieved context + question, in the _SEP format."""
        if chunks:
            context = "\n\n".join(format_chunk_for_context(c, i) for i, c in enumerate(chunks))
        else:
            context = "[No relevant sections found in this law.]"
        return "\n\n".join([
            self.system_prompt,
            _SEP,
            f"CONTEXT (from the {self.law}):\n\n{context}",
            _SEP,
            f"QUESTION:\n{question}",
            _SEP,
            "ANSWER:",
        ])

    def answer(self, question: str, top_k: int = 5) -> dict:
        """Retrieve within this law, then generate a cited answer."""
        chunks = self.retriever.search_scoped(question, self.law, top_k=top_k)
        prompt = self._build_prompt(question, chunks)
        answer = self.generator.generate(prompt)
        return {
            "law": self.law,
            "answer": answer,
            "sources": extract_sources_from_chunks(chunks),
            "chunks_used": len(chunks),
        }


def build_specialists(retriever, generator) -> dict:
    """Convenience: one shared retriever/generator -> {law: LawAgent}."""
    return {law: LawAgent(law, retriever, generator) for law in SPECIALIST_PROMPTS}


if __name__ == "__main__":
    # Smoke test — each specialist answers a question on its own turf.
    # Run: python3 agents/specialist.py
    from retriever import Retriever
    from generator import Generator

    retriever = Retriever()      # loaded ONCE, shared by all specialists
    generator = Generator()
    specialists = build_specialists(retriever, generator)

    probes = {
        "Constitution": "What does Article 21 guarantee?",
        "BNS": "What is the BNS equivalent of IPC Section 420?",
        "BNSS": "How do I file an FIR?",
    }
    for law, q in probes.items():
        print(f"\n{'='*70}\n  {law} specialist  <-  {q}\n{'='*70}")
        out = specialists[law].answer(q)
        print(out["answer"][:600])
        print(f"\n  sources: {[(s.get('law'), s.get('section') or s.get('article')) for s in out['sources']]}")
