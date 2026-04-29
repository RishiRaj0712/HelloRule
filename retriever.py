"""
retriever.py — LawBook India
=============================
Loads the ChromaDB vector store (once at startup) and retrieves
the top-K most relevant chunks for any user query.

Used by: main.py → pipeline → retriever.search()
"""

import os
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

    # BNS
    "murder":          {"law": "BNS"},
    "theft":           {"law": "BNS"},
    "rape":            {"law": "BNS"},
    "punishment for":  {"law": "BNS"},
    "bail":            {"law": "BNS"},
    "cheating":        {"law": "BNS"},
    "robbery":         {"law": "BNS"},
    "kidnapping":      {"law": "BNS"},
    "defamation":      {"law": "BNS"},
    "terrorism":       {"law": "BNS"},
    "organised crime": {"law": "BNS"},
    "stalking":        {"law": "BNS"},
    "dowry":           {"law": "BNS"},
    "sedition":        {"law": "BNS"},
}


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
        print(f"[Retriever] Ready — {count} chunks loaded from {CHROMA_DIR}/")

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

    def search_with_fallback(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Same as search() but if the filtered search returns fewer than
        top_k results (e.g. Part III only has 3 matching chunks),
        it falls back to an unfiltered search to fill the gap.

        This prevents the LLM from getting too little context.
        """
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