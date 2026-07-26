"""
eval_coverage.py — LawBook India: Systematic Ground-Truth Coverage Eval
==========================================================================
Runs every case from build_coverage_cases.py through the live /api/chat
endpoint and scores retrieval accuracy, citation accuracy, and faithfulness
(see eval_lib.py) — giving per-law, per-layer accuracy numbers instead of
just the ~34 hand-picked cases in test_suite.py.

Self-paces against the Groq free-tier rate limit and checkpoints every
result, so it's safe to interrupt and resume. If the limit keeps getting
hit even after cooldowns, it automatically downgrades the remaining queue
to the smaller stratified sample rather than stalling indefinitely.

Usage:
    python3 eval_coverage.py --mode full            # attempt all ~1480 cases (default)
    python3 eval_coverage.py --mode sample          # just the ~60-case stratified sample
    python3 eval_coverage.py --resume               # continue an interrupted run
    python3 eval_coverage.py --rpm 25                # override pacing budget
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import eval_lib

# All eval inputs/outputs live alongside this script, so it can be run from anywhere.
EVAL_DIR     = Path(__file__).resolve().parent
CASES_FULL   = str(EVAL_DIR / "eval_cases_full.json")
CASES_SAMPLE = str(EVAL_DIR / "eval_cases_sample20.json")
CHECKPOINT   = str(EVAL_DIR / "eval_progress.json")
REPORT       = str(EVAL_DIR / "eval_report.json")

DEFAULT_RPM = 8    # conservative — 25 RPM and then 12 RPM both still triggered sustained
                    # Groq 429s (likely a TPM, not just RPM, cap given how token-heavy legal
                    # answers are); the circuit breaker below is the real safety net regardless
CIRCUIT_BREAK_STREAK = 3       # consecutive rate-limited cases before first cooldown
WINDOW_SIZE = 10               # rolling window for the intermittent-throttling trigger
WINDOW_THRESHOLD = 5           # if >=5 of the last 10 were throttled, that's sustained too,
                                # even if none of them happened to land 3-in-a-row
COOLDOWN_SCHEDULE = [60, 300, 300]  # seconds — escalating cooldowns before downgrading


def load_cases(mode: str) -> list:
    path = CASES_FULL if mode == "full" else CASES_SAMPLE
    if not Path(path).exists():
        raise SystemExit(f"{path} not found — run build_coverage_cases.py first.")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_one_case(case: dict) -> dict:
    """Call the API, then score retrieval / citation / faithfulness for one case."""
    resp = eval_lib.call_api(case["query"], top_k=5)

    if "error" in resp or eval_lib.is_rate_limited(resp):
        return {
            "case_id": case["case_id"], "law": case["law"], "entry_type": case["entry_type"],
            "query": case["query"], "rate_limited": eval_lib.is_rate_limited(resp),
            "error": resp.get("error") or resp.get("answer", ""),
        }

    answer = resp.get("answer", "")
    sources = resp.get("sources", [])

    retrieval_hit = any(s.get("source_id") == case["source_id"] for s in sources) \
        if any("source_id" in s for s in sources) else None
    # main.py's extract_sources_from_chunks doesn't currently expose source_id to the
    # API response, so fall back to matching on (law, type, number) against expected.
    if retrieval_hit is None:
        expected = case.get("expected_citation")
        if expected:
            retrieved_set = {eval_lib._normalize_source_entry(s) for s in sources}
            retrieval_hit = tuple(expected) in retrieved_set
        else:
            retrieval_hit = len(sources) > 0  # retrieval-only entries (preamble/note/overview)

    cited = eval_lib.parse_cited_sources(answer)
    expected_tuple = tuple(case["expected_citation"]) if case.get("expected_citation") else None
    citation = eval_lib.citation_accuracy(cited, sources, expected=expected_tuple)

    faithfulness = eval_lib.score_faithfulness(answer, case["ground_truth_text"])

    return {
        "case_id": case["case_id"], "law": case["law"], "entry_type": case["entry_type"],
        "query": case["query"], "rate_limited": False, "error": None,
        "retrieval_hit": retrieval_hit,
        "citation": citation,
        "faithfulness": faithfulness,
        "answer_preview": answer[:200],
    }


def run(mode: str, rpm: int, resume: bool):
    cases = load_cases(mode)
    checkpoint = eval_lib.load_checkpoint(CHECKPOINT) if resume else None

    if checkpoint and checkpoint.get("mode") == mode:
        results = checkpoint["results"]
        done_ids = {r["case_id"] for r in results}
        print(f"Resuming — {len(done_ids)} cases already done.")
    else:
        results = []
        done_ids = set()

    remaining = [c for c in cases if c["case_id"] not in done_ids]
    print(f"\n{'='*65}\n  LawBook India — Coverage Eval  [{mode}]\n"
          f"  {len(remaining)} cases to run ({len(done_ids)} already done)\n{'='*65}\n")

    min_interval = 60.0 / rpm
    consecutive_rate_limited = 0
    recent_throttled = []  # rolling window of True/False — catches INTERMITTENT throttling
                            # (e.g. every other call fails) that never forms a 3-in-a-row streak
    downgraded = False

    def is_throttled(r):
        # A 110s timeout with no other error text is almost always a rate-limited
        # call that just missed the window — treat it the same as an explicit 429.
        return bool(r["rate_limited"]) or (r.get("error") == f"Request timed out after 110s")

    i = 0
    while i < len(remaining):
        case = remaining[i]
        start = time.time()

        result = run_one_case(case)
        results.append(result)

        throttled = is_throttled(result)
        consecutive_rate_limited = consecutive_rate_limited + 1 if throttled else 0
        recent_throttled.append(throttled)
        recent_throttled = recent_throttled[-WINDOW_SIZE:]

        status = "RATE-LIMITED" if result["rate_limited"] else (
            "OK" if result.get("retrieval_hit") else "MISS"
        )
        print(f"  [{len(results):>5}/{len(cases)}] {case['law']:<14} {case['entry_type']:<10} {status:<14} {case['query'][:55]}")

        eval_lib.save_checkpoint(CHECKPOINT, {"mode": mode, "results": results, "timestamp": datetime.now().isoformat()})

        # ── Circuit breaker: sustained rate-limiting (either a hard streak,
        # or frequent-but-intermittent throttling within the rolling window) ──
        sustained = consecutive_rate_limited >= CIRCUIT_BREAK_STREAK or \
            (len(recent_throttled) == WINDOW_SIZE and sum(recent_throttled) >= WINDOW_THRESHOLD)
        if sustained:
            handled = False
            for cooldown_idx, cooldown in enumerate(COOLDOWN_SCHEDULE):
                print(f"\n  Rate limit hit {consecutive_rate_limited}x in a row — cooling down {cooldown}s "
                      f"(attempt {cooldown_idx+1}/{len(COOLDOWN_SCHEDULE)})...\n")
                time.sleep(cooldown)

                probe = run_one_case(case)  # retry the same case as a probe
                if not is_throttled(probe):
                    results[-1] = probe  # replace the rate-limited record with the real result
                    eval_lib.save_checkpoint(CHECKPOINT, {"mode": mode, "results": results, "timestamp": datetime.now().isoformat()})
                    consecutive_rate_limited = 0
                    recent_throttled = []
                    handled = True
                    print(f"  Recovered — resuming normal pace.\n")
                    break

            if not handled:
                if mode == "full" and not downgraded:
                    print(f"\n  Still rate-limited after {len(COOLDOWN_SCHEDULE)} cooldown cycles — "
                          f"downgrading remaining queue to the ~20/law sample.\n")
                    sample_cases = load_cases("sample")
                    done_ids = {r["case_id"] for r in results}
                    remaining = remaining[:i+1] + [c for c in sample_cases if c["case_id"] not in done_ids]
                    downgraded = True
                else:
                    print(f"\n  Still rate-limited after cooldowns — stopping here. "
                          f"Resume later with --resume once your Groq quota resets.\n")
                    break

        # ── Steady-state pacing ──
        elapsed = time.time() - start
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        i += 1

    write_report(results, mode, downgraded)


def write_report(results: list, mode: str, downgraded: bool):
    # "scored" = actually went through citation/faithfulness scoring — excludes
    # both rate-limited cases AND other errors (timeouts, connection drops, etc.)
    scored = [r for r in results if r.get("citation") is not None]

    by_law = {}
    for r in scored:
        by_law.setdefault(r["law"], []).append(r)

    print(f"\n{'='*65}\n  SUMMARY{'  (downgraded to sample mode mid-run)' if downgraded else ''}\n{'-'*65}")

    summary = {}
    for law, rs in by_law.items():
        retrieval_hits = sum(1 for r in rs if r.get("retrieval_hit"))
        cite_checks = [r for r in rs if r["citation"]["expected"] is not None]
        cite_hits = sum(1 for r in cite_checks if r["citation"]["expected_hit"])
        precisions = [r["citation"]["precision"] for r in rs if r["citation"]["precision"] is not None]
        embed_scores = [r["faithfulness"]["embedding_score"] for r in rs]
        judge_scores = [r["faithfulness"]["judge_score"] for r in rs if r["faithfulness"]["judge_score"] is not None]

        law_summary = {
            "total": len(rs),
            "retrieval_hit_rate": round(retrieval_hits / len(rs), 3) if rs else 0,
            "citation_recall": round(cite_hits / len(cite_checks), 3) if cite_checks else None,
            "citation_precision_avg": round(sum(precisions) / len(precisions), 3) if precisions else None,
            "faithfulness_embedding_avg": round(sum(embed_scores) / len(embed_scores), 3) if embed_scores else None,
            "faithfulness_judge_escalations": len(judge_scores),
            "faithfulness_judge_avg": round(sum(judge_scores) / len(judge_scores), 3) if judge_scores else None,
        }
        summary[law] = law_summary

        print(f"  {law:<14} n={law_summary['total']:<5} "
              f"retrieval={law_summary['retrieval_hit_rate']:.0%}  "
              f"citation_recall={law_summary['citation_recall']}  "
              f"citation_precision={law_summary['citation_precision_avg']}  "
              f"faithfulness(embed)={law_summary['faithfulness_embedding_avg']}  "
              f"judge_escalations={law_summary['faithfulness_judge_escalations']} (avg={law_summary['faithfulness_judge_avg']})")

    unscored_count = len(results) - len(scored)
    print(f"{'-'*65}\n  Total cases run: {len(results)}  |  Unscored (errors/rate-limited): {unscored_count}")
    print(f"{'='*65}\n")

    report = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "downgraded": downgraded,
        "total_cases": len(results),
        "unscored": unscored_count,
        "by_law": summary,
        "results": results,
    }
    Path(REPORT).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  Full report saved to {REPORT}\n")


def main():
    parser = argparse.ArgumentParser(description="LawBook India — Coverage Eval")
    parser.add_argument("--mode", choices=["full", "sample"], default="full")
    parser.add_argument("--rpm", type=int, default=DEFAULT_RPM)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    try:
        health = __import__("requests").get(f"{eval_lib.API_BASE}/health", timeout=5).json()
        if not health.get("retriever") or not health.get("generator"):
            print("Backend not fully loaded. Start uvicorn first.")
            return
    except Exception:
        print(f"Cannot reach backend at {eval_lib.API_BASE} — run: uvicorn main:app --port 8000")
        return

    run(args.mode, args.rpm, args.resume)


if __name__ == "__main__":
    main()
