"""
synthesizer.py — The Synthesizer (fan-in / reduce step)  [Phase 3]
===================================================================
When a question is routed to more than one law, each specialist returns its
own answer. The synthesizer merges those into a SINGLE coherent, cited answer.

This is the "reduce" half of fan-out/fan-in. It is only invoked when there are
2+ specialist answers — a single answer is passed through untouched (the fast
path), so we never pay for an extra LLM call we don't need.

Design note — the synthesizer must NOT add new facts. Its only job is to
combine and organise what the specialists already said (each of which is
already grounded in retrieved text). Keeping the synthesizer "closed-book"
this way stops a second hallucination surface from opening up.
"""

# Reuse generator.py's system/user split convention (see specialist.py).
_SEP = "=" * 60

SYNTH_SYSTEM_PROMPT = """You are a legal-answer synthesizer for an Indian-law Q&A system. \
You are given a user's question and answers from one or more SPECIALIST agents, each an \
expert in a different law:
- Constitution of India
- BNS (Bharatiya Nyaya Sanhita) — criminal offences and punishments
- BNSS (Bharatiya Nagarik Suraksha Sanhita) — criminal procedure

Combine the specialist answers into ONE clear, coherent response.

Rules:
- Use ONLY what the specialist answers contain. Do NOT add facts, sections, or articles \
they did not mention.
- Preserve every citation exactly as given.
- Remove redundancy. If the answer spans an offence and its procedure, structure it clearly \
(e.g. "The offence — ... The procedure — ...").
- Write in plain English for a non-lawyer.
- End with a single consolidated "Sources used:" line covering all cited provisions."""


class Synthesizer:
    """Merges multiple specialist answers. Shares the injected Generator."""

    def __init__(self, generator):
        self.generator = generator

    def synthesize(self, question: str, results: list[dict]) -> str:
        """
        results: list of {"law", "answer", "sources", ...} from specialists.
        Returns one merged answer string.
        """
        blocks = [f"[{r['law']} specialist answered]:\n{r['answer']}" for r in results]
        prompt = "\n\n".join([
            SYNTH_SYSTEM_PROMPT,
            _SEP,
            "SPECIALIST ANSWERS:\n\n" + "\n\n".join(blocks),
            _SEP,
            f"USER QUESTION:\n{question}",
            _SEP,
            "SYNTHESIZED ANSWER:",
        ])
        return self.generator.generate(prompt)
