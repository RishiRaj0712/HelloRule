"""
eval_multiagent.py — Multi-Agent System Evaluation  [Phase 4]
==============================================================
The honest capstone: does the multi-agent pipeline actually BEAT the simple
single-pipeline app, and is its routing any good? Two evaluations:

  PART A — Router (component eval): tested in isolation, in-process. Does the
           supervisor pick the right law(s)? Multi-label, so we report exact-
           match accuracy AND per-label precision/recall.

  PART B — A/B (system eval): the SAME questions through both /api/chat
           (baseline) and /api/chat/multiagent (multi-agent), scoring
           retrieval hit-rate, citation precision, and latency. Holding the
           questions constant isolates the effect of the architecture.

Kept to a small curated set on purpose — this is a comparison, not a coverage
sweep, and the multi-agent path spends several LLM calls per question.

Usage:
    python3 evals/eval_multiagent.py
"""

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import eval_lib
from agents.router import Router, LAWS

# ── Curated cases ──────────────────────────────────────────────────────────
# laws   = the correct routing (a SET; empty = out of scope)
# expect = ground-truth (law, type, number) for retrieval scoring, or None
#          (multi-law / out-of-scope cases are routing-only).
CASES = [
    {"q": "What does Article 21 guarantee?",                    "laws": {"Constitution"}, "expect": ("Constitution", "article", "21")},
    {"q": "What does Article 14 say about equality?",           "laws": {"Constitution"}, "expect": ("Constitution", "article", "14")},
    {"q": "What does Article 44 say about a uniform civil code?","laws": {"Constitution"}, "expect": ("Constitution", "article", "44")},
    {"q": "What is the punishment for murder?",                 "laws": {"BNS"},          "expect": ("BNS", "section", "103")},
    {"q": "What is the punishment for cheating?",               "laws": {"BNS"},          "expect": ("BNS", "section", "318")},
    {"q": "What is the BNS equivalent of IPC Section 420?",     "laws": {"BNS"},          "expect": ("BNS", "section", "318")},
    {"q": "How do I file an FIR?",                              "laws": {"BNSS"},         "expect": ("BNSS", "section", "173")},
    {"q": "What is anticipatory bail?",                         "laws": {"BNSS"},         "expect": ("BNSS", "section", "482")},
    {"q": "Can police arrest someone without a warrant?",       "laws": {"BNSS"},         "expect": ("BNSS", "section", "35")},
    {"q": "What is CrPC Section 41 in the new law?",            "laws": {"BNSS"},         "expect": ("BNSS", "section", "35")},
    # Multi-law (routing + latency only)
    {"q": "For murder, what is the offence and how is the case tried?",   "laws": {"BNS", "BNSS"}, "expect": None},
    {"q": "For cheating, what is the offence and how is bail handled?",   "laws": {"BNS", "BNSS"}, "expect": None},
    # Out of scope
    {"q": "What is the GST rate on gold?",                      "laws": set(),            "expect": None},
    {"q": "How do I register a private limited company?",       "laws": set(),            "expect": None},
]


# ── PART A: Router component eval ──────────────────────────────────────────

def eval_routing():
    print(f"\n{'='*72}\n  PART A — Router (component eval)\n{'='*72}")
    router = Router()

    exact = 0
    # per-label tallies for precision/recall
    tp = {l: 0 for l in LAWS}; fp = {l: 0 for l in LAWS}; fn = {l: 0 for l in LAWS}

    for c in CASES:
        pred = set(router.route(c["q"])["laws"])
        gold = c["laws"]
        is_exact = pred == gold
        exact += is_exact
        for l in LAWS:
            if l in pred and l in gold: tp[l] += 1
            elif l in pred and l not in gold: fp[l] += 1
            elif l not in pred and l in gold: fn[l] += 1
        mark = "ok " if is_exact else "MISS"
        print(f"  [{mark}] pred={sorted(pred)!s:<28} gold={sorted(gold)!s:<28} {c['q'][:40]}")
        time.sleep(1)

    print(f"\n  Exact-match routing accuracy: {exact}/{len(CASES)} = {exact/len(CASES):.0%}")
    print(f"  {'label':<14}{'precision':>11}{'recall':>9}")
    for l in LAWS:
        p = tp[l] / (tp[l] + fp[l]) if (tp[l] + fp[l]) else 1.0
        r = tp[l] / (tp[l] + fn[l]) if (tp[l] + fn[l]) else 1.0
        print(f"  {l:<14}{p:>11.0%}{r:>9.0%}")


# ── PART B: A/B system eval ────────────────────────────────────────────────

def score_one(resp, expect):
    """retrieval_hit + citation precision for a single API response."""
    if "error" in resp:
        return {"hit": None, "precision": None, "error": resp["error"]}
    sources = resp.get("sources", [])
    answer = resp.get("answer", "")
    retrieved = {eval_lib._normalize_source_entry(s) for s in sources}
    hit = (tuple(expect) in retrieved) if expect else None
    cited = eval_lib.parse_cited_sources(answer)
    prec = eval_lib.citation_accuracy(cited, sources)["precision"]
    return {"hit": hit, "precision": prec, "error": None}


def eval_ab():
    print(f"\n{'='*72}\n  PART B — Baseline vs Multi-agent (system A/B)\n{'='*72}")
    ab_cases = [c for c in CASES if c["expect"] is not None]

    agg = {"base": {"hits": 0, "prec": [], "lat": []},
           "multi": {"hits": 0, "prec": [], "lat": []}}
    n = len(ab_cases)

    print(f"  {'question':<44}{'baseline':>16}{'multi-agent':>16}")
    print(f"  {'':<44}{'hit  prec  s':>16}{'hit  prec  s':>16}")
    for c in ab_cases:
        t0 = time.time(); base = eval_lib.call_api(c["q"], top_k=5);            lat_b = time.time() - t0
        t1 = time.time(); multi = eval_lib.call_api_multiagent(c["q"], top_k=5); lat_m = time.time() - t1
        sb = score_one(base, c["expect"]); sm = score_one(multi, c["expect"])

        for tag, s, lat in [("base", sb, lat_b), ("multi", sm, lat_m)]:
            if s["hit"]: agg[tag]["hits"] += 1
            if s["precision"] is not None: agg[tag]["prec"].append(s["precision"])
            agg[tag]["lat"].append(lat)

        def cell(s, lat):
            h = "Y" if s["hit"] else ("-" if s["hit"] is None else "N")
            p = f"{s['precision']:.2f}" if s["precision"] is not None else " -- "
            return f"{h:>3} {p:>5} {lat:>4.1f}"
        print(f"  {c['q'][:44]:<44}{cell(sb, lat_b):>16}{cell(sm, lat_m):>16}")
        time.sleep(1)

    def summarize(tag):
        a = agg[tag]
        prec = sum(a["prec"]) / len(a["prec"]) if a["prec"] else 0
        lat = sum(a["lat"]) / len(a["lat"]) if a["lat"] else 0
        return a["hits"], prec, lat

    hb, pb, lb = summarize("base")
    hm, pm, lm = summarize("multi")
    print(f"\n  {'metric':<26}{'baseline':>12}{'multi-agent':>14}{'delta':>10}")
    print(f"  {'retrieval hit-rate':<26}{hb}/{n:>10}{f'{hm}/{n}':>14}{f'{hm-hb:+d}':>10}")
    print(f"  {'citation precision':<26}{pb:>12.0%}{pm:>14.0%}{f'{(pm-pb)*100:+.0f} pts':>10}")
    print(f"  {'avg latency (s)':<26}{lb:>12.1f}{lm:>14.1f}{f'{lm-lb:+.1f}s':>10}")


if __name__ == "__main__":
    try:
        h = __import__("requests").get(f"{eval_lib.API_BASE}/health", timeout=5).json()
        assert h.get("retriever")
    except Exception:
        print(f"Backend not reachable at {eval_lib.API_BASE} — start uvicorn first.")
        sys.exit(1)

    eval_routing()
    eval_ab()
    print()
