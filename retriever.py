"""
retriever.py — LawBook India
=============================
Loads the ChromaDB vector store (once at startup) and retrieves
the top-K most relevant chunks for any user query.

Used by: main.py → pipeline → retriever.search()
"""

import os
import re
import chromadb
from chromadb.utils import embedding_functions

# ── Must match ingest.py exactly ──
CHROMA_DIR  = "./chroma_db"
COLLECTION  = "constitution_of_india"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Keywords mapping
# Used for automatic metadata filtering when query mentions a specific keyword
KEYWORDS = {
    # Constitution
    "fundamental right":  {"part": "III"},
    "directive principle": {"part": "IV"},
    "fundamental duty":   {"part": "IVA"},
    "citizenship":        {"part": "II"},
    "emergency":          {"part": "XVIII"},
    "union territory":    {"part": "I"},
    "parliament":         {"part": "V"},
    "president":          {"part": "V"},
    "governor":           {"part": "VI"},
    "high court":         {"part": "VI"},
    "supreme court":      {"part": "V"},
    "finance":            {"part": "XII"},
    "trade":              {"part": "XIII"},
    "election":           {"part": "XV"},
    "language":           {"part": "XVII"},
    "official language":  {"part": "XVII"},
    "make laws":          {"part": "XI"},
    "national emergency": {"part": "XVIII"},
    "declare emergency":  {"part": "XVIII"},

    # BNS (substantive criminal law — offences & punishments)
    "murder":          {"law": "BNS"},
    "theft":           {"law": "BNS"},
    "rape":            {"law": "BNS"},
    "punishment for":  {"law": "BNS"},
    "cheating":        {"law": "BNS"},
    "robbery":         {"law": "BNS"},
    "kidnapping":      {"law": "BNS"},
    "defamation":      {"law": "BNS"},
    "terrorism":       {"law": "BNS"},
    "organised crime": {"law": "BNS"},
    "stalking":        {"law": "BNS"},
    "dowry":           {"law": "BNS"},
    "sedition":        {"law": "BNS"},

    # BNSS (procedural criminal law — process, courts, bail, investigation)
    "bail":              {"law": "BNSS"},
    "arrest":            {"law": "BNSS"},
    "fir":               {"law": "BNSS"},
    "warrant":           {"law": "BNSS"},
    "summons":           {"law": "BNSS"},
    "investigation":     {"law": "BNSS"},
    "cognizable":        {"law": "BNSS"},
    "non-cognizable":    {"law": "BNSS"},
    "magistrate":        {"law": "BNSS"},
    "trial":             {"law": "BNSS"},
    "appeal":            {"law": "BNSS"},
    "plea bargaining":   {"law": "BNSS"},
    "bail bond":         {"law": "BNSS"},
    "anticipatory bail": {"law": "BNSS"},
    "charge sheet":      {"law": "BNSS"},
    "police station":    {"law": "BNSS"},
    "zero fir":          {"law": "BNSS"},
    "e-fir":             {"law": "BNSS"},
    "remand":            {"law": "BNSS"},
    "custody":           {"law": "BNSS"},
    "search and seizure":{"law": "BNSS"},
    "confession":        {"law": "BNSS"},
    "witness":           {"law": "BNSS"},
    "sentence":          {"law": "BNSS"},
    "mercy petition":    {"law": "BNSS"},
    "compounding":       {"law": "BNSS"},
}


# Direct numeric-citation patterns ("Article 21", "Section 103 of the BNS",
# "BNSS Section 173"). Semantic search alone frequently retrieves the wrong
# provision for these — hundreds of BNS/BNSS sections use similar wording,
# so a query for an exact number needs an exact metadata match, not a
# similarity ranking. See _detect_exact_reference().
_ARTICLE_NUM_RE     = re.compile(r'\bArticle\s+(\d{1,3}[A-Za-z]{0,2})\b', re.IGNORECASE)
_LAW_SECTION_NUM_RE = re.compile(r'\b(BNS|BNSS)\s+Section\s+(\d{1,4}[A-Za-z]{0,2})\b', re.IGNORECASE)
_SECTION_OF_LAW_RE  = re.compile(r'\bSection\s+(\d{1,4}[A-Za-z]{0,2})\s+of\s+(?:the\s+)?(BNS|BNSS)\b', re.IGNORECASE)
_BARE_SECTION_RE    = re.compile(r'\bSection\s+(\d{1,4}[A-Za-z]{0,2})\b', re.IGNORECASE)
_AMENDMENT_ORDINAL_RE = re.compile(r'\b(\d{1,3})(?:st|nd|rd|th)?\s+Amendment\b', re.IGNORECASE)
_SCHEDULE_ORDINAL_RE  = re.compile(r'\b(\d{1,2})(?:st|nd|rd|th)?\s+Schedule\b', re.IGNORECASE)

# Old-law references ("IPC 420", "IPC Section 420", "Section 302 of the IPC").
# The number here is an OLD-law section — it must be TRANSLATED to the new
# BNS/BNSS section, never matched literally (IPC 420 is BNS 318, not BNSS 420).
# These must be detected before the bare-section match, which would otherwise
# grab the old number and look it up directly in the new law.
_IPC_REF_RE  = re.compile(
    r'\b(?:IPC|Indian\s+Penal\s+Code)\s+(?:Section\s+)?(\d{1,4}[A-Za-z]{0,2})\b'
    r'|\bSection\s+(\d{1,4}[A-Za-z]{0,2})\s+of\s+(?:the\s+)?(?:IPC|Indian\s+Penal\s+Code)\b',
    re.IGNORECASE)
_CRPC_REF_RE = re.compile(
    r'\b(?:CrPC|Code\s+of\s+Criminal\s+Procedure|Criminal\s+Procedure\s+Code)\s+(?:Section\s+)?(\d{1,4}[A-Za-z]{0,2})\b'
    r'|\bSection\s+(\d{1,4}[A-Za-z]{0,2})\s+of\s+(?:the\s+)?(?:CrPC|Code\s+of\s+Criminal\s+Procedure|Criminal\s+Procedure\s+Code)\b',
    re.IGNORECASE)

# Repealed-and-replaced sections that the auto-built maps miss: the new
# section carries no "ipc_equivalent"/"crpc_equivalent" back-reference to the
# old number (because the offence was rewritten, not renumbered), so it must
# be aliased explicitly. Keys are UPPERCASE old-law numbers.
#   IPC 124A (sedition) was repealed; BNS 152 is its closest replacement.
# (IPC 377 is intentionally NOT here — it has no BNS replacement, so it falls
#  through to semantic search, which surfaces the "not included in BNS" note.)
_IPC_ALIAS  = {"124A": "152"}
_CRPC_ALIAS = {}


class Retriever:
    """
    Wraps ChromaDB with a clean .search() interface.
    Instantiate once at app startup — loading the embedding model
    takes ~3 seconds and should not happen on every request.
    """

    def __init__(self):
        print("[Retriever] Loading embedding model and ChromaDB...")

        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )

        client = chromadb.PersistentClient(path=CHROMA_DIR)

        self.collection = client.get_collection(
            name=COLLECTION,
            embedding_function=self.embed_fn,
        )

        count = self.collection.count()

        # Build old-law -> new-law translation maps from the section metadata
        # (ipc_equivalent / crpc_equivalent). Used to turn "IPC 420" into the
        # correct BNS section (318) rather than matching 420 literally.
        self.ipc_to_bns, self.crpc_to_bnss = self._build_translation_maps()
        print(f"[Retriever] Ready — {count} chunks loaded from {CHROMA_DIR}/ "
              f"({len(self.ipc_to_bns)} IPC→BNS, {len(self.crpc_to_bnss)} CrPC→BNSS mappings)")

    def _build_translation_maps(self) -> tuple[dict, dict]:
        """
        Reverse-map old-law section numbers to new-law sections using each
        entry's ipc_equivalent / crpc_equivalent metadata. Only unambiguous
        1:1 mappings are kept — old numbers that map to several new sections
        (e.g. the omnibus 'Definitions' section) fall back to filtered
        semantic search instead.
        """
        from collections import defaultdict

        ipc_map  = defaultdict(set)
        crpc_map = defaultdict(set)

        data = self.collection.get(include=["metadatas"])
        for meta in data["metadatas"]:
            if meta.get("type") != "section":
                continue
            section = str(meta.get("section", "")).strip()
            if not section:
                continue
            for old in str(meta.get("ipc_equivalent", "")).split(","):
                old = old.strip()
                if old:
                    ipc_map[old].add(section)
            for old in str(meta.get("crpc_equivalent", "")).split(","):
                old = old.strip()
                if old:
                    crpc_map[old].add(section)

        ipc_single  = {k: next(iter(v)) for k, v in ipc_map.items() if len(v) == 1}
        crpc_single = {k: next(iter(v)) for k, v in crpc_map.items() if len(v) == 1}

        # Fill gaps for repealed-and-replaced sections the metadata can't map.
        for old, new in _IPC_ALIAS.items():
            ipc_single.setdefault(old, new)
        for old, new in _CRPC_ALIAS.items():
            crpc_single.setdefault(old, new)

        return ipc_single, crpc_single

    def _detect_filters(self, query: str) -> dict | None:
        """
        Optionally narrow the search space using metadata filters.

        If the query clearly mentions a specific Part or type
        (e.g. "fundamental rights" → Part III), we pre-filter
        before similarity search — faster and more precise.

        Returns a ChromaDB 'where' clause or None (no filtering).
        """
        query_lower = query.lower()

        # Explicit law mentions
        if "under bns" in query_lower or "bharatiya nyaya sanhita" in query_lower or "criminal law" in query_lower:
            return {"law": "BNS"}
        if "under the constitution" in query_lower or "constitution of india" in query_lower:
            return {"law": "Constitution"}
        if "under bnss" in query_lower or "bharatiya nagarik suraksha" in query_lower or "nagarik suraksha sanhita" in query_lower:
            return {"law": "BNSS"}

        # CrPC translation -> BNSS
        # Since BNSS chunks contain "[Old CrPC: X]", filtering to law=BNSS allows the vector search to find it.
        if "crpc" in query_lower or "criminal procedure code" in query_lower or "code of criminal procedure" in query_lower:
            return {"law": "BNSS"}

        # IPC translation
        # e.g. "IPC 302" -> translate to BNS search
        # Since BNS chunks contain "[Old IPC: 302]", filtering to law=BNS allows the vector search to find it.
        if "ipc" in query_lower or "indian penal code" in query_lower:
            return {"law": "BNS"}

        # Check for amendment-specific queries
        if any(w in query_lower for w in ["amendment", "amended", "amend the constitution"]):
            return {"type": "amendment"}

        # Check for schedule-specific queries
        if "schedule" in query_lower:
            return {"type": "schedule"}

        # Check for keywords
        for keyword, filters in KEYWORDS.items():
            if keyword in query_lower:
                return filters

        # No filter — search all chunks
        return None

    def _detect_exact_reference(self, query: str) -> dict | None:
        """
        Detect a literal article/section number in the query and build an
        exact ChromaDB metadata filter for it. Returns None if no direct
        numeric reference is found (or if an old-law reference can't be
        translated — those defer to filtered semantic search).
        """
        # ── Old-law references first ("IPC 420", "CrPC Section 154") ──
        # The number is an OLD section that must be TRANSLATED, not matched
        # literally. If we have a clean 1:1 mapping, exact-match the new
        # section; otherwise return None so the query falls through to a
        # law-filtered semantic search (which can still find it via the
        # "[Old IPC: N]" text embedded in each chunk). Either way we must
        # NOT let the bare-section match below grab the old number.
        m = _IPC_REF_RE.search(query)
        if m:
            old = (m.group(1) or m.group(2)).upper()
            new_section = self.ipc_to_bns.get(old)
            if new_section:
                return {"$and": [{"law": "BNS"}, {"section": new_section.upper()}]}
            return None

        m = _CRPC_REF_RE.search(query)
        if m:
            old = (m.group(1) or m.group(2)).upper()
            new_section = self.crpc_to_bnss.get(old)
            if new_section:
                return {"$and": [{"law": "BNSS"}, {"section": new_section.upper()}]}
            return None

        m = _ARTICLE_NUM_RE.search(query)
        if m:
            return {"$and": [{"law": "Constitution"}, {"article": m.group(1).upper()}]}

        m = _LAW_SECTION_NUM_RE.search(query)
        if m:
            return {"$and": [{"law": m.group(1).upper()}, {"section": m.group(2).upper()}]}

        m = _SECTION_OF_LAW_RE.search(query)
        if m:
            return {"$and": [{"law": m.group(2).upper()}, {"section": m.group(1).upper()}]}

        # "Section 103" with no law named — try BNS/BNSS both (Constitution
        # uses "Article", never "Section", so it's excluded here).
        m = _BARE_SECTION_RE.search(query)
        if m:
            return {"$and": [{"law": {"$in": ["BNS", "BNSS"]}}, {"section": m.group(1).upper()}]}

        m = _AMENDMENT_ORDINAL_RE.search(query)
        if m:
            return {"$and": [{"law": "Constitution"}, {"article": f"Amendment {int(m.group(1))}"}]}

        # Schedules are stored as "Schedule N" or, for the 7th Schedule's
        # three sub-lists, "Schedule N — Union/State/Concurrent List" — an
        # exact-equality where filter can't do a prefix match, so this is
        # flagged for special handling in _exact_match().
        m = _SCHEDULE_ORDINAL_RE.search(query)
        if m:
            return {"_schedule_num": int(m.group(1))}

        return None

    def _exact_match(self, where_filter: dict) -> list[dict]:
        """
        Direct metadata lookup, no similarity ranking — every chunk of the
        matched entry, ordered by chunk_index. Used for direct numeric
        citations where semantic search is unreliable.
        """
        if "_schedule_num" in where_filter:
            # Schedule article values aren't a clean equality match (see note
            # above) — fetch all schedules and prefix-match in Python, being
            # careful that "Schedule 1" doesn't also match "Schedule 10"-"19".
            prefix = f"Schedule {where_filter['_schedule_num']}"
            results = self.collection.get(
                where={"$and": [{"type": "schedule"}, {"law": "Constitution"}]},
                include=["documents", "metadatas"],
            )
            chunks = []
            for i in range(len(results["ids"])):
                meta = results["metadatas"][i]
                article = str(meta.get("article", ""))
                if article == prefix or article.startswith(prefix + " "):
                    chunks.append({"text": results["documents"][i], "metadata": meta, "score": 1.0})
            chunks.sort(key=lambda c: int(c["metadata"].get("chunk_index", 0)))
            return chunks

        results = self.collection.get(where=where_filter, include=["documents", "metadatas"])

        chunks = []
        for i in range(len(results["ids"])):
            chunks.append({
                "text":     results["documents"][i],
                "metadata": results["metadatas"][i],
                "score":    1.0,  # exact match, not a similarity score
            })

        chunks.sort(key=lambda c: int(c["metadata"].get("chunk_index", 0)))
        return chunks

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Find the top-K chunks most relevant to the query.

        Args:
            query:  The user's question (raw, unmodified)
            top_k:  How many chunks to return (default 5)

        Returns:
            List of dicts, each containing:
              - text:     The full chunk text (context prefix + content)
              - metadata: article, part, title, type, status, etc.
              - score:    Cosine similarity (0-1, higher = more relevant)
        """
        where_filter = self._detect_filters(query)

        query_params = {
            "query_texts": [query],
            "n_results":   top_k,
            "include":     ["documents", "metadatas", "distances"],
        }

        if where_filter:
            query_params["where"] = where_filter

        results = self.collection.query(**query_params)

        chunks = []
        for i in range(len(results["ids"][0])):
            chunks.append({
                "text":     results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score":    round(1 - results["distances"][0][i], 4),
            })

        return chunks

    def search_scoped(self, query: str, law: str, top_k: int = 5) -> list[dict]:
        """
        Retrieve ONLY within a single law — used by the multi-agent specialists
        (agents/specialist.py), where the supervisor has already decided the
        law and each specialist must stay in its lane.

        This is a NEW method (the existing app keeps using search_with_fallback
        unchanged) — an additive, backward-compatible extension.

        It still benefits from the exact-match / IPC-CrPC translation logic,
        but only when that resolves to THIS law; otherwise it does a plain
        law-filtered semantic search.
        """
        # Exact/translation first — but keep only hits belonging to `law`.
        # (For a correctly-routed question these already match; the filter is
        # a safety net against, e.g., a bare "Section 5" resolving to the other
        # criminal-law book.)
        exact_filter = self._detect_exact_reference(query)
        if exact_filter:
            exact_chunks = [c for c in self._exact_match(exact_filter)
                            if c["metadata"].get("law") == law]
            if exact_chunks:
                return exact_chunks[:top_k]

        # Law-scoped semantic search — the hard {"law": law} filter is what
        # makes this a "specialist" retrieval.
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where={"law": law},
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for i in range(len(results["ids"][0])):
            chunks.append({
                "text":     results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score":    round(1 - results["distances"][0][i], 4),
            })
        return chunks

    def search_with_fallback(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Same as search() but if the filtered search returns fewer than
        top_k results (e.g. Part III only has 3 matching chunks),
        it falls back to an unfiltered search to fill the gap.

        This prevents the LLM from getting too little context.
        """
        # ── Exact numeric citation ("Article 21", "Section 103 of the BNS") ──
        # Try this before semantic search — with hundreds of similarly-worded
        # BNS/BNSS sections, similarity ranking alone frequently picks the
        # wrong one for a direct number lookup.
        exact_filter = self._detect_exact_reference(query)
        if exact_filter:
            exact_chunks = self._exact_match(exact_filter)
            if exact_chunks:
                if len(exact_chunks) < top_k:
                    exact_ids = {(c["metadata"].get("source_id"), c["metadata"].get("chunk_index")) for c in exact_chunks}
                    filler = self.search(query, top_k=top_k)
                    for c in filler:
                        key = (c["metadata"].get("source_id"), c["metadata"].get("chunk_index"))
                        if key not in exact_ids and len(exact_chunks) < top_k:
                            exact_chunks.append(c)
                return exact_chunks[:top_k]
            # No hit (e.g. article number doesn't exist / was repealed and
            # dropped from the dataset) — fall through to normal search below.

        where_filter = self._detect_filters(query)

        if where_filter:
            # Try filtered first
            filtered_results = self.search(query, top_k=top_k)
            if len(filtered_results) >= 3:
                return filtered_results
            # Not enough — fall back to unfiltered
            print(f"[Retriever] Filter {where_filter} returned {len(filtered_results)} results — falling back to unfiltered")

        # Unfiltered search
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for i in range(len(results["ids"][0])):
            chunks.append({
                "text":     results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score":    round(1 - results["distances"][0][i], 4),
            })

        return chunks