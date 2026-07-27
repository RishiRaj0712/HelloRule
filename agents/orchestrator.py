"""
orchestrator.py — The Orchestrator (fan-out/fan-in coordinator)  [Phase 3]
===========================================================================
Ties the multi-agent system together:

    router.route(question)          -> which law(s)?
      │
      ├─ []            -> polite out-of-scope reply (no LLM calls at all)
      ├─ [one law]     -> that specialist answers  (fast path, no synthesis)
      └─ [many laws]   -> specialists run IN PARALLEL, then synthesizer merges

Concepts shown here:
  • Fan-out/fan-in — scatter to specialists, gather + reduce.
  • Concurrency — specialists run in a ThreadPoolExecutor (I/O-bound LLM calls
    overlap, cutting multi-law latency roughly in half).
  • Graceful degradation — if one specialist errors, the others still answer.
  • Doing the least work — out-of-scope and single-law paths skip the extra
    LLM call that synthesis would cost.
"""

import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.router import Router, LAWS            # noqa: E402
from agents.specialist import build_specialists   # noqa: E402
from agents.synthesizer import Synthesizer        # noqa: E402

OUT_OF_SCOPE_MSG = (
    "This question doesn't appear to fall under the Constitution of India, the BNS "
    "(criminal offences), or the BNSS (criminal procedure) — the three laws I cover. "
    "Please consult the relevant law or a legal expert."
)


class Orchestrator:
    """Coordinates router + specialists + synthesizer. Build once, reuse."""

    def __init__(self, router, specialists: dict, synthesizer, max_workers: int = 3):
        self.router = router
        self.specialists = specialists
        self.synthesizer = synthesizer
        self.max_workers = max_workers

    def _merge_sources(self, results: list[dict]) -> list[dict]:
        """Flatten + de-duplicate sources across specialists, preserving order."""
        seen, merged = set(), []
        for r in results:
            for s in r.get("sources", []):
                key = (s.get("law"), s.get("article") or s.get("section"))
                if key not in seen:
                    seen.add(key)
                    merged.append(s)
        return merged

    def answer(self, question: str, top_k: int = 5) -> dict:
        # 1. ROUTE
        routing = self.router.route(question)
        laws = routing["laws"]

        # 2. OUT OF SCOPE — no specialist, no LLM answer call
        if not laws:
            return {
                "answer": OUT_OF_SCOPE_MSG,
                "sources": [],
                "routing": routing,
                "agents_used": [],
                "synthesized": False,
            }

        # 3. FAN-OUT — run the chosen specialists in parallel.
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_to_law = {
                pool.submit(self.specialists[law].answer, question, top_k): law
                for law in laws if law in self.specialists
            }
            for future in as_completed(future_to_law):
                law = future_to_law[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    # Graceful degradation: log the failed specialist, keep going.
                    results.append({"law": law, "answer": f"[{law} specialist failed: {e}]",
                                    "sources": [], "chunks_used": 0, "error": True})

        # Restore the router's law order (as_completed yields out of order).
        order = {law: i for i, law in enumerate(LAWS)}
        results.sort(key=lambda r: order.get(r["law"], 99))
        ok_results = [r for r in results if not r.get("error")]

        # 4. FAN-IN — synthesize only if more than one specialist answered.
        if len(ok_results) <= 1:
            final_answer = (ok_results[0]["answer"] if ok_results
                            else "All specialists failed to answer. Please try again.")
            synthesized = False
        else:
            final_answer = self.synthesizer.synthesize(question, ok_results)
            synthesized = True

        return {
            "answer": final_answer,
            "sources": self._merge_sources(ok_results),
            "routing": routing,
            "agents_used": [r["law"] for r in results],
            "synthesized": synthesized,
        }


if __name__ == "__main__":
    # Smoke test — run: python3 agents/orchestrator.py
    from retriever import Retriever
    from generator import Generator

    retriever = Retriever()
    generator = Generator()
    router = Router()
    specialists = build_specialists(retriever, generator)
    synthesizer = Synthesizer(generator)
    orch = Orchestrator(router, specialists, synthesizer)

    for q in [
        "What does Article 21 guarantee?",                                   # single law
        "For murder, what is the offence and how is the case tried?",        # multi-law
        "What is the GST rate on gold?",                                     # out of scope
    ]:
        print(f"\n{'='*72}\n  Q: {q}\n{'='*72}")
        out = orch.answer(q)
        print(f"  routed to : {out['routing']['laws']}  ({out['routing']['reasoning']})")
        print(f"  agents    : {out['agents_used']}  | synthesized: {out['synthesized']}")
        print(f"  sources   : {[(s.get('law'), s.get('section') or s.get('article')) for s in out['sources']]}")
        print(f"\n{out['answer'][:700]}")
