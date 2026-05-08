"""
test_retrieval.py — Verify your ChromaDB vector store is working correctly.
============================================================================
Run this AFTER ingest.py has completed successfully.

Tests 10 real-world queries across different parts of the Constitution,
shows retrieved chunks with similarity scores, and flags any quality issues.

Usage:
    python test_retrieval.py
    python test_retrieval.py --query "Can police arrest without warrant?"
    python test_retrieval.py --top_k 3
"""

import argparse
import chromadb
from chromadb.utils import embedding_functions

# ── Must match ingest.py exactly ──
CHROMA_DIR = "./chroma_db"
COLLECTION = "constitution_of_india"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# ─────────────────────────────────────────────
# TEST QUERIES — Constitution + BNS + BNSS
# ─────────────────────────────────────────────
# Each has: the user's informal question + what we EXPECT to retrieve
TEST_QUERIES = [
    # ── CONSTITUTION (6 queries) ──
    {
        "query": "What are the fundamental rights of Indian citizens?",
        "expect": "Part III / Articles 12-35",
        "why": "Broad query — should pull multiple Part III chunks",
    },
    {
        "query": "How can the Indian Constitution be amended?",
        "expect": "Article 368",
        "why": "Tests retrieval of a specific amendment procedure article",
    },
    {
        "query": "What powers does the President have during an emergency?",
        "expect": "Article 352 / 356 / 360",
        "why": "Tests emergency provisions retrieval (Part XVIII)",
    },
    {
        "query": "What is the right to education?",
        "expect": "Article 21A",
        "why": "Tests an amendment-inserted article (86th Amendment 2002)",
    },
    {
        "query": "What are the Directive Principles of State Policy?",
        "expect": "Part IV / Articles 36-51",
        "why": "Tests a broad Part-level query",
    },
    {
        "query": "When was untouchability abolished in India?",
        "expect": "Article 17",
        "why": "Specific single-article query — should rank very high",
    },

    # ── BNS — Substantive Criminal Law (4 queries) ──
    {
        "query": "What is the punishment for murder in India?",
        "expect": "BNS Section 103",
        "why": "Core BNS offence — murder definition and punishment",
    },
    {
        "query": "What is IPC Section 302 in the new law?",
        "expect": "BNS Section 103 [Old IPC: 302]",
        "why": "IPC-to-BNS translation — should find BNS via ipc_equivalent tag",
    },
    {
        "query": "Is sedition still a crime in India?",
        "expect": "BNS Section 152 / sedition removed note",
        "why": "Sedition was removed — should find the note entry",
    },
    {
        "query": "What is the offence of theft under BNS?",
        "expect": "BNS Section 303/304",
        "why": "Tests BNS keyword filter routing",
    },

    # ── BNSS — Criminal Procedure (10 queries) ──
    {
        "query": "What is the procedure for bail in India?",
        "expect": "BNSS Section 478-483 (Bail provisions)",
        "why": "Tests BNSS bail routing — should NOT return BNS results",
    },
    {
        "query": "Can police arrest someone without a warrant?",
        "expect": "BNSS Section 35 (Arrest without warrant)",
        "why": "Tests arrest procedure — BNSS keyword filter",
    },
    {
        "query": "How to file a Zero FIR?",
        "expect": "BNSS Section 173 / Zero FIR note",
        "why": "Key BNSS reform — Zero FIR provision",
    },
    {
        "query": "What is CrPC Section 154 in the new law?",
        "expect": "BNSS Section 173 [Old CrPC: 154]",
        "why": "CrPC-to-BNSS translation — should find BNSS via crpc_equivalent tag",
    },
    {
        "query": "What is anticipatory bail?",
        "expect": "BNSS Section 482 [Old CrPC: 438]",
        "why": "Tests anticipatory bail retrieval from BNSS",
    },
    {
        "query": "What is the procedure for plea bargaining?",
        "expect": "BNSS Section 289-300 (Chapter XXIII)",
        "why": "Tests plea bargaining chapter retrieval",
    },
    {
        "query": "Can trials be held in electronic mode?",
        "expect": "BNSS Section 530 / Electronic mode note",
        "why": "New BNSS provision — electronic proceedings",
    },
    {
        "query": "What is the maximum time police can detain someone without producing before a magistrate?",
        "expect": "BNSS Section 58 [Old CrPC: 57]",
        "why": "24-hour detention rule — key procedural safeguard",
    },
    {
        "query": "What is the procedure for investigation of a cognizable offence?",
        "expect": "BNSS Section 173-196 (Chapter XIII)",
        "why": "Tests investigation procedure chapter retrieval",
    },
    {
        "query": "How is a mercy petition handled in death sentence cases?",
        "expect": "BNSS Section 472 (Mercy petition timeline)",
        "why": "Key BNSS reform — mercy petition timeline",
    },
]

def load_collection():
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        collection = client.get_collection(
            name=COLLECTION,
            embedding_function=embed_fn,
        )
    except Exception:
        print(f"ERROR: Collection '{COLLECTION}' not found in {CHROMA_DIR}/")
        print("  Did you run ingest.py first?")
        exit(1)

    return collection

def run_query(collection, query: str, top_k: int = 5) -> dict:
    """Run a single query and return structured results."""
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
            "score": round(1 - results["distances"][0][i], 4),  # cosine similarity
        })

    return chunks

def print_query_result(query_info: dict, chunks: list, show_full_text: bool = False):
    """Pretty-print the results for one test query."""
    print(f"\n{'═'*60}")
    print(f"  QUERY  : {query_info['query']}")
    print(f"  EXPECT : {query_info['expect']}")
    print(f"  WHY    : {query_info['why']}")
    print(f"{'─'*60}")

    for rank, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        score = chunk["score"]
        article = meta.get("article", "?")
        title = meta.get("title", "")
        part = meta.get("part", "")
        ctype = meta.get("type", "")
        status = meta.get("status", "active")
        is_multi = meta.get("total_chunks", "1") != "1"

        # Flag non-active articles in results
        status_flag = "" if meta.get("is_active") == "True" else f"  ⚠ [{status[:40]}]"

        section = meta.get("section", "")
        law = meta.get("law", "")
        if ctype == "article":
            label = f"Art.{article}"
        elif ctype == "section" and law:
            label = f"{law} Sec.{section}"
        else:
            label = ctype.upper()
        if is_multi:
            label += f" (chunk {meta.get('chunk_index','?')}/{int(meta.get('total_chunks',1))-1})"

        bar_len = int(score * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)

        print(f"  #{rank}  [{bar}] {score:.4f}  {label} — {title[:45]}{status_flag}")
        print(f"      Part {part or 'N/A'} | type={ctype}")

        if show_full_text:
            # Show first 300 chars of the chunk text
            preview = chunk["text"][:300].replace("\n", " ")
            print(f"      TEXT: {preview}...")

    # Quality assessment
    top_score = chunks[0]["score"] if chunks else 0
    if top_score >= 0.55:
        verdict = "✅ GOOD  — top result looks relevant"
    elif top_score >= 0.40:
        verdict = "⚠  FAIR  — top result may be relevant, review manually"
    else:
        verdict = "❌ POOR  — retrieval might be off, check chunking"

    print(f"\n  Verdict: {verdict}  (top score: {top_score:.4f})")

def run_all_tests(collection, top_k: int = 5):
    """Run the full test suite and print a summary."""
    print(f"\n{'═'*60}")
    print(f"  LawBook India — Retrieval Quality Test")
    print(f"  Collection: {COLLECTION}  |  Chunks: {collection.count()}")
    print(f"  Embedding model: {EMBED_MODEL}")
    print(f"{'═'*60}")

    scores = []
    for test in TEST_QUERIES:
        chunks = run_query(collection, test["query"], top_k=top_k)
        print_query_result(test, chunks, show_full_text=False)
        if chunks:
            scores.append(chunks[0]["score"])

    # Summary
    if scores:
        avg_score = sum(scores) / len(scores)
        good = sum(1 for s in scores if s >= 0.55)
        fair = sum(1 for s in scores if 0.40 <= s < 0.55)
        poor = sum(1 for s in scores if s < 0.40)

        print(f"\n{'═'*60}")
        print(f"  SUMMARY — {len(TEST_QUERIES)} test queries")
        print(f"{'─'*60}")
        print(f"  Avg top score : {avg_score:.4f}")
        print(f"  ✅ Good  (≥0.55) : {good}")
        print(f"  ⚠  Fair  (≥0.40) : {fair}")
        print(f"  ❌ Poor  (<0.40) : {poor}")
        print(f"{'═'*60}\n")

        if poor > 0:
            print("  ACTION: Some queries retrieved poor results.")
            print("  Consider: larger context prefixes, different chunk boundaries,")
            print("  or upgrading to 'nomic-embed-text' for better legal English.")
        elif fair > 0:
            print("  ACTION: Most queries look good. Review 'fair' results manually.")
        else:
            print("  All queries retrieved relevant results. Ready for Phase 3!")

def run_single_query(collection, query: str, top_k: int = 5):
    """Run a single custom query interactively."""
    chunks = run_query(collection, query, top_k=top_k)
    test_info = {"query": query, "expect": "user-defined", "why": "manual test"}
    print_query_result(test_info, chunks, show_full_text=True)

def main():
    parser = argparse.ArgumentParser(description="LawBook India — Test Retrieval Quality")
    parser.add_argument("--query", default=None, help="Run a single custom query")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to return")
    args = parser.parse_args()

    collection = load_collection()

    if args.query:
        run_single_query(collection, args.query, top_k=args.top_k)
    else:
        run_all_tests(collection, top_k=args.top_k)

if __name__ == "__main__":
    main()