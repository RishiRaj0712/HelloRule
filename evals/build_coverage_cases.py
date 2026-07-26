"""
build_coverage_cases.py — LawBook India: Ground-Truth Eval Case Generator
============================================================================
Reads the three law JSON files and auto-generates test cases with a
deterministic, known-correct answer (source_id + expected citation) — used
by eval_coverage.py to systematically check retrieval/citation/faithfulness
across ALL entries, not just hand-picked cases.

Writes two files (inspect them before spending API quota on a full run):
  eval_cases_full.json     — one case per entry (~1480 cases)
  eval_cases_sample20.json — stratified ~20/law fallback sample

Usage:
    python3 build_coverage_cases.py
"""

import json
import random
import re
from pathlib import Path

# Anchor paths so the script works regardless of the current directory:
# law data lives at the repo root, generated cases live alongside this script.
REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent

LAW_FILES = {
    "Constitution": REPO_ROOT / "data" / "constitution_final.json",
    "BNS": REPO_ROOT / "data" / "bns_final.json",
    "BNSS": REPO_ROOT / "data" / "bnss_final.json",
}

SAMPLE_PER_LAW = 20
RANDOM_SEED = 42


def ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def build_case(entry: dict, law: str) -> dict | None:
    """One entry -> one eval case, or None if it can't produce a sensible query."""
    description = (entry.get("description") or "").strip()
    if not description:
        return None

    entry_type = entry.get("type", "article")
    entry_id = entry.get("id", "")
    title = entry.get("title", "")

    if entry_type == "article":
        num = str(entry.get("article", "")).strip()
        if not num:
            return None
        query = f"What does Article {num} of the Indian Constitution say?"
        expected_citation = ["Constitution", "article", num.upper()]

    elif entry_type == "section":
        num = str(entry.get("section", "")).strip()
        if not num:
            return None
        query = f"What does Section {num} of the {law} say?"
        expected_citation = [law, "section", num.upper()]

    elif entry_type == "amendment":
        raw = str(entry.get("article", ""))  # e.g. "Amendment 21"
        m = re.search(r'(\d+)', raw)
        if not m:
            return None
        num = int(m.group(1))
        query = f"What did the {ordinal(num)} Amendment to the Indian Constitution do?"
        expected_citation = ["Constitution", "amendment", str(num)]

    elif entry_type == "schedule":
        raw = str(entry.get("article", ""))  # e.g. "Schedule 1" or "Schedule 7 -- Union List"
        m = re.search(r'(\d+)', raw)
        if not m:
            return None
        num = int(m.group(1))
        topic = entry.get("topic", "") or title
        query = f"What does the {ordinal(num)} Schedule of the Indian Constitution cover? ({topic})"
        expected_citation = ["Constitution", "schedule", str(num)]

    elif entry_type == "preamble":
        query = "What does the Preamble of the Indian Constitution say?"
        expected_citation = None  # no clean citation number — retrieval-only check

    elif entry_type in ("overview", "note"):
        # No clean citation number — these test retrieval of important edge-case
        # entries (e.g. "sedition removed") rather than citation accuracy.
        if not title:
            return None
        query = f"Tell me about: {title}"
        expected_citation = None

    else:
        return None

    return {
        "case_id": entry_id,
        "law": law,
        "entry_type": entry_type,
        "source_id": entry_id,
        "expected_citation": expected_citation,
        "query": query,
        "ground_truth_text": description,
    }


def stratified_sample(cases_by_type: dict, target: int, seed: int) -> list:
    """
    Sample ~target cases from a law's cases, spread across entry types
    proportionally, but always including every entry from small/rare types
    (preamble, overview, note) since those cover important edge cases.
    """
    rng = random.Random(seed)
    sample = []

    small_types = {"preamble", "overview", "note"}
    large_types = [t for t in cases_by_type if t not in small_types]

    for t in cases_by_type:
        if t in small_types:
            sample.extend(cases_by_type[t])

    remaining_slots = max(target - len(sample), 0)
    if large_types and remaining_slots > 0:
        per_type = max(remaining_slots // len(large_types), 1)
        for t in large_types:
            pool = cases_by_type[t]
            n = min(per_type, len(pool))
            sample.extend(rng.sample(pool, n))

    return sample


def main():
    all_cases = []
    cases_by_law_type = {}

    for law, path in LAW_FILES.items():
        if not path.exists():
            print(f"WARNING: {path} not found, skipping {law}")
            continue

        entries = json.loads(path.read_text(encoding="utf-8"))
        law_cases_by_type = {}
        for entry in entries:
            case = build_case(entry, law)
            if case is None:
                continue
            all_cases.append(case)
            law_cases_by_type.setdefault(case["entry_type"], []).append(case)

        cases_by_law_type[law] = law_cases_by_type

        print(f"{law:<14} {len(entries):>4} entries -> {sum(len(v) for v in law_cases_by_type.values()):>4} cases  "
              f"({', '.join(f'{t}:{len(v)}' for t, v in law_cases_by_type.items())})")

    (EVAL_DIR / "eval_cases_full.json").write_text(json.dumps(all_cases, indent=2), encoding="utf-8")
    print(f"\nWrote eval_cases_full.json — {len(all_cases)} total cases")

    sample_cases = []
    for law, by_type in cases_by_law_type.items():
        law_sample = stratified_sample(by_type, SAMPLE_PER_LAW, RANDOM_SEED)
        sample_cases.extend(law_sample)
        print(f"  {law} sample: {len(law_sample)} cases")

    (EVAL_DIR / "eval_cases_sample20.json").write_text(json.dumps(sample_cases, indent=2), encoding="utf-8")
    print(f"Wrote eval_cases_sample20.json — {len(sample_cases)} total cases")


if __name__ == "__main__":
    main()
