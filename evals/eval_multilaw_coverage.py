"""
eval_multilaw_coverage.py — Baseline vs Multi-agent on MULTI-LAW questions
===========================================================================
Phase 4's A/B only scored single-provision questions, so it couldn't measure
the one thing the multi-agent design is actually built for: questions that
span TWO laws (e.g. "murder — the offence AND how it's tried").

Hypothesis: the baseline retrieves top-K from the whole mixed pool, so one law
can crowd out the other; the multi-agent retrieves top-K from EACH routed law,
so it should cover both sides.

Metric — dual-law coverage:
  • retrieval coverage: do the returned sources include >=1 provision from
    EVERY required law?
  • citation coverage: does the ANSWER actually cite >=1 provision from every
    required law?

Usage:
    python3 evals/eval_multilaw_coverage.py
"""

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import eval_lib

# Multi-law questions and the set of laws a good answer MUST cover.
CASES = [
    {"q": "For murder, what is the offence and how is the case tried?",           "need": {"BNS", "BNSS"}},
    {"q": "For cheating, what is the offence and how is bail handled?",           "need": {"BNS", "BNSS"}},
    {"q": "How is theft punished and how is such a case investigated?",           "need": {"BNS", "BNSS"}},
    {"q": "What is the punishment for kidnapping and how is bail granted for it?", "need": {"BNS", "BNSS"}},
]


def coverage(resp: dict, need: set) -> dict:
    """laws present in sources / in the answer's citations, vs. what's needed."""
    if "error" in resp:
        return {"src_laws": set(), "cite_laws": set(), "src_ok": False, "cite_ok": False,
                "per_law_counts": {}, "error": resp["error"]}

    sources = resp.get("sources", [])
    answer = resp.get("answer", "")

    src_laws = {s.get("law") for s in sources}
    per_law_counts = {}
    for s in sources:
        per_law_counts[s.get("law")] = per_law_counts.get(s.get("law"), 0) + 1

    cite_laws = {c[0] for c in eval_lib.parse_cited_sources(answer)}

    return {
        "src_laws": src_laws,
        "cite_laws": cite_laws,
        "src_ok": need.issubset(src_laws),
        "cite_ok": need.issubset(cite_laws),
        "per_law_counts": per_law_counts,
        "error": None,
    }


def main():
    try:
        h = __import__("requests").get(f"{eval_lib.API_BASE}/health", timeout=5).json()
        assert h.get("retriever")
    except Exception:
        print(f"Backend not reachable at {eval_lib.API_BASE} — start uvicorn first.")
        sys.exit(1)

    print(f"\n{'='*78}\n  Multi-law coverage — Baseline vs Multi-agent\n{'='*78}")
    print("  (src = >=1 source from every required law | cite = answer cites every required law)\n")

    agg = {"base": {"src": 0, "cite": 0}, "multi": {"src": 0, "cite": 0}}
    n = len(CASES)

    for c in CASES:
        base = eval_lib.call_api(c["q"], top_k=5)
        multi = eval_lib.call_api_multiagent(c["q"], top_k=5)
        cb = coverage(base, c["need"])
        cm = coverage(multi, c["need"])

        agg["base"]["src"] += cb["src_ok"];   agg["base"]["cite"] += cb["cite_ok"]
        agg["multi"]["src"] += cm["src_ok"];   agg["multi"]["cite"] += cm["cite_ok"]

        print(f"  Q: {c['q']}")
        print(f"     need: {sorted(c['need'])}")
        print(f"     baseline    — sources by law: {cb['per_law_counts']}  "
              f"covered={cb['src_ok']}  cited-both={cb['cite_ok']}")
        print(f"     multi-agent — sources by law: {cm['per_law_counts']}  "
              f"covered={cm['src_ok']}  cited-both={cm['cite_ok']}\n")
        time.sleep(1)

    base_src  = f"{agg['base']['src']}/{n}";   multi_src  = f"{agg['multi']['src']}/{n}"
    base_cite = f"{agg['base']['cite']}/{n}";  multi_cite = f"{agg['multi']['cite']}/{n}"
    print(f"{'-'*78}")
    print(f"  {'metric':<28}{'baseline':>12}{'multi-agent':>14}")
    print(f"  {'retrieval covers both laws':<28}{base_src:>12}{multi_src:>14}")
    print(f"  {'answer cites both laws':<28}{base_cite:>12}{multi_cite:>14}")
    print(f"{'='*78}\n")


if __name__ == "__main__":
    main()
