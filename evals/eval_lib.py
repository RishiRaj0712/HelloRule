"""
eval_lib.py — LawBook India: Shared Evaluation Utilities
==========================================================
Building blocks used by eval_coverage.py (and reusable by any future eval
script) to check response *authenticity* — not just keyword presence:

  1. Citation parsing  — what did the model actually CITE in its answer text
                          (separate from what was merely RETRIEVED)
  2. Citation accuracy — precision (no invented citations) + recall (expected
                          ground-truth citation was actually used)
  3. Faithfulness      — does the answer's substance match the real legal text?
                          Fast embedding-similarity pass, escalated to an
                          LLM-judge call only when the score is borderline/low.

Nothing in this file makes network calls at import time — models/clients are
loaded lazily on first use.
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re
import json
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

# .env lives at the repo root (one level up from this evals/ folder)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

API_BASE = "http://localhost:8000"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # must match ingest.py / retriever.py
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama-3.1-8b-instant")

# Below this embedding-similarity score, escalate to the LLM judge.
FAITHFULNESS_ESCALATE_THRESHOLD = 0.55


# ─────────────────────────────────────────────
# API CALLS
# ─────────────────────────────────────────────

def call_api(query: str, history: list = None, top_k: int = 5, timeout: int = 110) -> dict:
    """
    Call the /api/chat endpoint. Same pattern as test_suite.py's call_api.

    Timeout must comfortably exceed generator.py's own worst case: on a
    rate limit it retries 3x internally with 15s/30s/45s sleeps (~90s)
    before giving up and returning its fallback string. A shorter client
    timeout doesn't avoid that cost — the backend keeps retrying regardless
    — it just means the client gives up early, sees a generic "timed out"
    instead of the real rate-limit response, and eval_coverage.py's circuit
    breaker (which pattern-matches on the rate-limit text) never fires.
    """
    for attempt in range(2):
        try:
            res = requests.post(
                f"{API_BASE}/api/chat",
                json={"query": query, "history": history or [], "top_k": top_k},
                timeout=timeout,
            )
            res.raise_for_status()
            return res.json()
        except requests.exceptions.ConnectionError:
            if attempt == 0:
                time.sleep(3)
                continue
            return {"error": "Cannot connect to backend. Is uvicorn running on port 8000?"}
        except requests.exceptions.Timeout:
            return {"error": f"Request timed out after {timeout}s"}
        except Exception as e:
            return {"error": str(e)}


def is_rate_limited(response: dict) -> bool:
    """
    Detect generator.py's own exhausted-retry fallback ("Rate limit reached...")
    or a raw 429 surfaced in the answer/error text. Used by eval_coverage.py's
    batch-level circuit breaker (generator.py already retries once internally;
    this catches the case where even THAT was exhausted).
    """
    text = ""
    if "error" in response:
        text = str(response["error"])
    text += " " + str(response.get("answer", ""))
    text = text.lower()
    return "rate limit" in text or "429" in text or "too many requests" in text


# ─────────────────────────────────────────────
# 1. CITATION PARSING
# ─────────────────────────────────────────────
# Extracts what the MODEL actually wrote in its answer — e.g. "Article 21",
# "BNS Section 103", "21st Amendment", "Seventh Schedule" — as
# (law, entry_type, number) triples. This is intentionally separate from the
# API's `sources` field, which reflects retrieval, not what the model chose
# to cite.
#
# entry_type is tracked deliberately: "21st Amendment" and "Article 21" must
# NOT collide into the same key just because both reduce to "21" — they are
# different legal provisions, and conflating them is exactly the kind of
# mistake this eval exists to catch (see the real Article-21/21st-Amendment
# mix-up found during manual testing).

_ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11,
    "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
}

_ARTICLE_RE        = re.compile(r'\bArticle\s+(\d{1,3}[A-Za-z]{0,2})\b', re.IGNORECASE)
_LAW_SECTION_RE     = re.compile(r'\b(BNS|BNSS)\s+Section\s+(\d{1,4}[A-Za-z]{0,2})\b', re.IGNORECASE)
_SECTION_OF_LAW_RE  = re.compile(r'\bSection\s+(\d{1,4}[A-Za-z]{0,2})\s+of\s+(?:the\s+)?(BNS|BNSS)\b', re.IGNORECASE)

_AMENDMENT_ORDINAL_RE = re.compile(r'\b(\d{1,3})(?:st|nd|rd|th)?\s+Amendment\b', re.IGNORECASE)
_AMENDMENT_ALT_RE     = re.compile(r'\bAmendment\s+(\d{1,3})\b', re.IGNORECASE)

_SCHEDULE_NUM_RE   = re.compile(r'\b(\d{1,2})(?:st|nd|rd|th)?\s+Schedule\b', re.IGNORECASE)
_SCHEDULE_ALT_RE   = re.compile(r'\bSchedule\s+(\d{1,2})\b', re.IGNORECASE)
_SCHEDULE_WORD_RE  = re.compile(
    r'\b(' + '|'.join(_ORDINAL_WORDS.keys()) + r')\s+Schedule\b', re.IGNORECASE
)


def parse_cited_sources(answer_text: str) -> set:
    """Returns a set of (law, entry_type, number) triples, e.g. {("Constitution", "article", "21")}."""
    cited = set()

    for m in _ARTICLE_RE.finditer(answer_text):
        cited.add(("Constitution", "article", m.group(1).upper()))

    for m in _LAW_SECTION_RE.finditer(answer_text):
        cited.add((m.group(1).upper(), "section", m.group(2).upper()))

    for m in _SECTION_OF_LAW_RE.finditer(answer_text):
        cited.add((m.group(2).upper(), "section", m.group(1).upper()))

    for m in _AMENDMENT_ORDINAL_RE.finditer(answer_text):
        cited.add(("Constitution", "amendment", m.group(1)))
    for m in _AMENDMENT_ALT_RE.finditer(answer_text):
        cited.add(("Constitution", "amendment", m.group(1)))

    for m in _SCHEDULE_NUM_RE.finditer(answer_text):
        cited.add(("Constitution", "schedule", m.group(1)))
    for m in _SCHEDULE_ALT_RE.finditer(answer_text):
        cited.add(("Constitution", "schedule", m.group(1)))
    for m in _SCHEDULE_WORD_RE.finditer(answer_text):
        num = _ORDINAL_WORDS.get(m.group(1).lower())
        if num:
            cited.add(("Constitution", "schedule", str(num)))

    return cited


_PREFIX_STRIP_RE = re.compile(r'^(Amendment|Schedule)\s+', re.IGNORECASE)


def _normalize_source_entry(s: dict) -> tuple:
    """(law, entry_type, bare_number) for one entry from the API's `sources` list."""
    law = s.get("law", "Constitution")
    entry_type = s.get("type", "article")
    raw_num = str(s.get("article") or s.get("section") or "")
    num = _PREFIX_STRIP_RE.sub("", raw_num).strip().upper()
    return (law, entry_type, num)


# ─────────────────────────────────────────────
# 2. CITATION ACCURACY
# ─────────────────────────────────────────────

def citation_accuracy(cited: set, retrieved_sources: list, expected: tuple = None) -> dict:
    """
    cited:             set of (law, entry_type, number) the model actually wrote — from parse_cited_sources()
    retrieved_sources: the API response's `sources` list (what was actually retrieved)
    expected:          (law, entry_type, number) ground-truth triple this query was built to test, or None

    Returns precision (cited numbers actually grounded in retrieval — catches
    invented citations) and expected_hit (did the model cite the correct one).
    """
    retrieved_set = {_normalize_source_entry(s) for s in retrieved_sources if (s.get("article") or s.get("section"))}

    grounded = cited & retrieved_set
    invented = cited - retrieved_set
    precision = round(len(grounded) / len(cited), 4) if cited else None

    expected_hit = None
    if expected:
        expected_norm = (expected[0], expected[1], str(expected[2]).upper())
        expected_hit = expected_norm in cited

    return {
        "cited": sorted(cited),
        "retrieved": sorted(retrieved_set),
        "invented": sorted(invented),
        "precision": precision,
        "expected": list(expected) if expected else None,
        "expected_hit": expected_hit,
    }


# ─────────────────────────────────────────────
# 3. FAITHFULNESS — embedding pass + LLM-judge escalation
# ─────────────────────────────────────────────

_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


def embedding_faithfulness(answer_text: str, ground_truth_text: str) -> float:
    """Cosine similarity between the answer and the real legal text it should be grounded in."""
    from sentence_transformers import util
    model = _get_embed_model()
    emb = model.encode([answer_text, ground_truth_text], normalize_embeddings=True)
    return round(float(util.cos_sim(emb[0], emb[1])[0][0]), 4)


_groq_client = None

JUDGE_PROMPT = """You are a strict legal fact-checker. Compare the ANSWER against the CONTEXT \
(the actual legal text it should be grounded in). Judge ONLY whether the ANSWER's factual claims \
are supported by the CONTEXT — ignore style, tone, or missing detail.

CONTEXT (ground truth):
{context}

ANSWER TO CHECK:
{answer}

Rate 1-5:
  5 = every factual claim is fully supported by the context
  3 = mostly supported, minor unsupported details
  1 = contradicts the context or is mostly unsupported

Respond with ONLY a JSON object, no other text:
{{"score": <int 1-5>, "unsupported_claims": ["...", "..."]}}
"""


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq_client


def llm_judge_faithfulness(answer_text: str, context_text: str, max_retries: int = 3) -> dict:
    """Escalation path — only called when embedding_faithfulness() is below threshold."""
    client = _get_groq_client()
    prompt = JUDGE_PROMPT.format(context=context_text[:3000], answer=answer_text[:2000])

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=300,
            )
            raw = resp.choices[0].message.content.strip()
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return {
                    "score": int(data.get("score", 0)),
                    "unsupported_claims": data.get("unsupported_claims", []),
                    "raw": raw,
                }
            return {"score": None, "unsupported_claims": [], "raw": raw, "error": "no JSON in judge response"}

        except Exception as e:
            msg = str(e)
            if any(k in msg.lower() for k in ["429", "rate_limit", "rate limit", "too many requests"]):
                if attempt < max_retries - 1:
                    time.sleep(15 * (attempt + 1))
                    continue
            return {"score": None, "unsupported_claims": [], "raw": "", "error": msg}

    return {"score": None, "unsupported_claims": [], "raw": "", "error": "rate limited after retries"}


def score_faithfulness(answer_text: str, ground_truth_text: str) -> dict:
    """
    Hybrid faithfulness check: embedding similarity first (free), LLM-judge
    only escalated when the embedding score is borderline/low.
    """
    embed_score = embedding_faithfulness(answer_text, ground_truth_text)

    result = {
        "embedding_score": embed_score,
        "escalated": False,
        "judge_score": None,
        "unsupported_claims": [],
    }

    if embed_score < FAITHFULNESS_ESCALATE_THRESHOLD:
        judge = llm_judge_faithfulness(answer_text, ground_truth_text)
        result["escalated"] = True
        result["judge_score"] = judge.get("score")
        result["unsupported_claims"] = judge.get("unsupported_claims", [])
        if judge.get("error"):
            result["judge_error"] = judge["error"]

    return result


# ─────────────────────────────────────────────
# CHECKPOINTING
# ─────────────────────────────────────────────

def save_checkpoint(path: str, data: dict):
    """Atomic write — avoids a corrupt checkpoint if interrupted mid-write."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def load_checkpoint(path: str):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)
