"""
test_translation.py — LawBook India: Old-Law Translation Eval
==============================================================
Tests the model on "old law -> new law" translation queries, e.g.
"What is the BNS equivalent of IPC Section 420?" or "What is CrPC 154 now?".

These are a distinct, tricky class from direct lookups ("What does BNS
Section 103 say?"): the number in the query (an IPC/CrPC section) is NOT
the answer's number — it must be translated to the corresponding BNS/BNSS
section via each entry's ipc_equivalent / crpc_equivalent metadata.

Ground truth is derived from the law JSON itself (not hand-typed), so the
expected answers are authoritative. Only IPC/CrPC numbers that map to a
SINGLE new-law section are used, to keep each case unambiguous.

Usage:
    python3 evals/test_translation.py            # run all cases
    python3 evals/test_translation.py --verbose  # show answer previews
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import eval_lib

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = {
    "BNS":  REPO_ROOT / "data" / "bns_final.json",
    "BNSS": REPO_ROOT / "data" / "bnss_final.json",
}

# A few well-known offences we explicitly want covered, plus a removed one.
WELL_KNOWN_IPC = ["420", "302", "379", "376", "300", "392", "120B", "499", "354", "363"]
WELL_KNOWN_CRPC = ["154", "41", "437", "438", "161", "164"]


def build_reverse_maps():
    """old-law number -> new-law section, keeping only unambiguous 1:1 mappings."""
    ipc_to_bns = defaultdict(set)
    crpc_to_bnss = defaultdict(set)

    bns = json.loads(DATA["BNS"].read_text(encoding="utf-8"))
    for e in bns:
        if e.get("type") == "section" and e.get("ipc_equivalent"):
            for ipc in str(e["ipc_equivalent"]).split(","):
                ipc_to_bns[ipc.strip()].add(str(e["section"]))

    bnss = json.loads(DATA["BNSS"].read_text(encoding="utf-8"))
    for e in bnss:
        if e.get("type") == "section" and e.get("crpc_equivalent"):
            for crpc in str(e["crpc_equivalent"]).split(","):
                crpc_to_bnss[crpc.strip()].add(str(e["section"]))

    # keep only clean single-target mappings
    ipc_single = {k: next(iter(v)) for k, v in ipc_to_bns.items() if len(v) == 1}
    crpc_single = {k: next(iter(v)) for k, v in crpc_to_bnss.items() if len(v) == 1}
    return ipc_single, crpc_single


def build_cases():
    ipc_single, crpc_single = build_reverse_maps()
    cases = []

    # IPC -> BNS (well-known offences with clean mappings)
    for ipc in WELL_KNOWN_IPC:
        if ipc in ipc_single:
            cases.append({
                "query": f"What is the BNS equivalent of IPC Section {ipc}?",
                "old": f"IPC {ipc}", "law": "BNS", "expected_section": ipc_single[ipc],
            })

    # A spread of other clean IPC mappings (sampled deterministically by sorting)
    extra_ipc = [k for k in sorted(ipc_single, key=lambda x: (len(x), x))
                 if k not in WELL_KNOWN_IPC][:10]
    for ipc in extra_ipc:
        cases.append({
            "query": f"Which BNS section replaced IPC {ipc}?",
            "old": f"IPC {ipc}", "law": "BNS", "expected_section": ipc_single[ipc],
        })

    # CrPC -> BNSS
    for crpc in WELL_KNOWN_CRPC:
        if crpc in crpc_single:
            cases.append({
                "query": f"What is the BNSS equivalent of CrPC Section {crpc}?",
                "old": f"CrPC {crpc}", "law": "BNSS", "expected_section": crpc_single[crpc],
            })

    # Special case: IPC 124A (sedition) was removed — replacement is BNS 152.
    cases.append({
        "query": "What is the BNS equivalent of IPC Section 124A (sedition)?",
        "old": "IPC 124A", "law": "BNS", "expected_section": "152",
        "note": "sedition removed; BNS 152 is the replacement offence",
    })

    return cases


def evaluate(case, resp):
    if "error" in resp:
        return {"retrieval_hit": False, "cited": False, "wrong_number": None, "error": resp["error"]}

    answer = resp.get("answer", "")
    sources = resp.get("sources", [])
    expected = (case["law"], case["expected_section"].upper())

    # Layer 1: was the correct new-law section retrieved at all?
    retrieved = {(s.get("law"), str(s.get("section") or "").upper()) for s in sources}
    retrieval_hit = expected in retrieved

    # Layer 2: did the model cite the correct section (law, type=section, number)?
    cited = eval_lib.parse_cited_sources(answer)
    cited_hit = (case["law"], "section", case["expected_section"].upper()) in cited

    # Failure signature: did it instead latch onto the OLD number as a new-law section?
    old_num = case["old"].split()[-1].upper()
    wrong_number = (case["law"], "section", old_num) in cited or \
        any(str(s.get("section") or "").upper() == old_num for s in sources)

    return {
        "retrieval_hit": retrieval_hit,
        "cited": cited_hit,
        "wrong_number": wrong_number,
        "error": None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    try:
        health = __import__("requests").get(f"{eval_lib.API_BASE}/health", timeout=5).json()
        assert health.get("retriever") and health.get("generator")
    except Exception:
        print(f"Backend not reachable at {eval_lib.API_BASE} — run: uvicorn main:app --port 8000")
        return

    cases = build_cases()
    print(f"\n{'='*70}\n  Old-Law Translation Eval — {len(cases)} cases\n{'='*70}\n")

    passed = wrong = 0
    for c in cases:
        resp = eval_lib.call_api(c["query"], top_k=5)
        r = evaluate(c, resp)

        ok = r["retrieval_hit"] and r["cited"]
        passed += ok
        wrong += bool(r["wrong_number"])
        mark = "PASS" if ok else "FAIL"
        flag = "  <- latched onto OLD number" if r["wrong_number"] else ""
        print(f"  [{mark}] {c['old']:>10} -> expect {c['law']} {c['expected_section']:<5} "
              f"| retrieved={r['retrieval_hit']!s:<5} cited={r['cited']!s:<5}{flag}")
        if args.verbose and not ok:
            print(f"         Q: {c['query']}")
            print(f"         A: {resp.get('answer','')[:180]}...")

    print(f"\n{'-'*70}")
    print(f"  Passed: {passed}/{len(cases)}   |   Latched onto old number: {wrong}/{len(cases)}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
