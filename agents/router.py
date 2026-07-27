"""
router.py — The Supervisor / Router agent  [Phase 1]
=====================================================
Looks at a user question and decides which law(s) it concerns:
Constitution, BNS (criminal offences), BNSS (criminal procedure) — one,
several, or none (out of scope).

This is an LLM-based intent classifier. We deliberately use the LLM (not the
existing keyword heuristic in retriever._detect_filters) because it handles
paraphrases, ambiguity, and — crucially — the MULTI-LAW case that makes the
downstream specialist/synthesizer design worthwhile.

Design notes:
  • Multi-label output (a list) so "offence AND procedure for murder" -> [BNS, BNSS].
  • Structured JSON with a `reasoning` field: asking the model to justify its
    pick first (a lightweight chain-of-thought) improves classification
    accuracy and gives a debuggable trace.
  • temperature=0 for near-deterministic routing.
"""

import os
import re
import json
from pathlib import Path

from groq import Groq
from dotenv import load_dotenv

# .env lives at the repo root (one level up from this agents/ folder)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

ROUTER_MODEL = os.getenv("ROUTER_MODEL", "llama-3.1-8b-instant")

# The canonical set of laws the system knows. Kept here so the router, the
# specialists, and the orchestrator all agree on the same labels.
LAWS = ["Constitution", "BNS", "BNSS"]

ROUTER_SYSTEM_PROMPT = """You are the routing supervisor for an Indian-law Q&A system. \
Your ONLY job is to decide which body of law a question concerns. You do NOT answer the \
question itself.

The system covers exactly three laws:
- "Constitution": the Constitution of India — fundamental rights, directive principles, \
government structure (Parliament, President, Governors, Judiciary), elections, emergency \
provisions, schedules, and amendments.
- "BNS": Bharatiya Nyaya Sanhita, 2023 — SUBSTANTIVE criminal law: what is an offence and \
its punishment (murder, theft, cheating, rape, cybercrime, etc.). Replaced the IPC.
- "BNSS": Bharatiya Nagarik Suraksha Sanhita, 2023 — criminal PROCEDURE: arrest, bail, FIR, \
investigation, trial, appeals, sentencing process, courts, warrants. Replaced the CrPC.

Rules:
- Return every law that is relevant. Many questions touch more than one (e.g. "what is the \
punishment for murder and how is such a case investigated?" -> both BNS and BNSS).
- IPC references map to BNS; CrPC references map to BNSS.
- If the question is outside all three laws (e.g. GST, contract law, civil procedure), \
return an empty list.

Respond with ONLY a JSON object, no other text:
{"laws": ["<one or more of Constitution|BNS|BNSS, or empty>"], "reasoning": "<one short sentence>"}"""

# A few worked examples steer the model toward the right granularity. Few-shot
# examples are one of the cheapest, highest-leverage ways to shape LLM behaviour.
FEW_SHOT = """Examples:
Q: "What does Article 21 guarantee?"
{"laws": ["Constitution"], "reasoning": "Fundamental right to life — constitutional."}
Q: "What is the punishment for cheating?"
{"laws": ["BNS"], "reasoning": "Substantive offence and punishment."}
Q: "How do I file an FIR?"
{"laws": ["BNSS"], "reasoning": "Criminal procedure — filing a first information report."}
Q: "For murder, what is the offence and how is the trial conducted?"
{"laws": ["BNS", "BNSS"], "reasoning": "Offence (BNS) plus trial procedure (BNSS)."}
Q: "What is the GST rate on gold?"
{"laws": [], "reasoning": "Tax law — outside all three covered laws."}"""


class Router:
    """LLM-based supervisor. Instantiate once; call .route() per question."""

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found.")
        self.client = Groq(api_key=api_key)
        self.model = ROUTER_MODEL
        print(f"[Router] Ready — using Groq/{self.model}")

    def route(self, question: str) -> dict:
        """
        Returns {"laws": [...], "reasoning": str}.
        `laws` is a validated subset of LAWS (possibly empty = out of scope).
        """
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ROUTER_SYSTEM_PROMPT + "\n\n" + FEW_SHOT},
                    {"role": "user", "content": f'Q: "{question}"'},
                ],
                temperature=0,
                max_tokens=150,
                # Groq/Llama supports forced JSON output — removes the "model
                # wrapped JSON in prose" failure mode. We still parse defensively.
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content.strip()
            return self._parse(raw)
        except Exception as e:
            # Fail safe: if routing breaks, fall back to "all laws" so the
            # question still gets answered (just less efficiently) rather than
            # dropped. Failing open vs. failing closed is a real design choice.
            return {"laws": list(LAWS), "reasoning": f"router error, defaulting to all: {e}"}

    def _parse(self, raw: str) -> dict:
        """Extract + validate the JSON, keeping only recognised law labels."""
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group(0)) if match else {}

        laws = data.get("laws", [])
        if not isinstance(laws, list):
            laws = [laws]
        # Normalise + validate against the canonical set (case-insensitive).
        valid = []
        lookup = {l.lower(): l for l in LAWS}
        for l in laws:
            key = str(l).strip().lower()
            if key in lookup and lookup[key] not in valid:
                valid.append(lookup[key])

        return {"laws": valid, "reasoning": data.get("reasoning", "")}


if __name__ == "__main__":
    # Quick manual smoke test — run: python3 agents/router.py
    r = Router()
    tests = [
        "What does Article 21 guarantee?",
        "What is the punishment for murder?",
        "Can police arrest someone without a warrant?",
        "What is the BNS equivalent of IPC Section 420?",
        "For murder, what is the offence and how is the case investigated and tried?",
        "What is the GST rate on gold?",
        "rights",
    ]
    print()
    for q in tests:
        out = r.route(q)
        print(f"  {str(out['laws']):<32} <- {q}")
        print(f"  {'':<32}    ({out['reasoning']})")
