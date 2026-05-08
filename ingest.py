"""
ingest.py — LawBook India RAG Pipeline: Phase 2
================================================
This script does 3 things in sequence:
  1. CHUNK  — Split 585 JSON entries into ~800 retrieval-sized pieces
  2. EMBED  — Convert each chunk to a 384-dim vector using bge-small-en-v1.5
  3. STORE  — Save everything into ChromaDB (persisted to ./chroma_db/)

Run this ONCE before starting the backend server.
Usage:
    python ingest.py
    python ingest.py --json path/to/constitution_final.json
    python ingest.py --reset   # clears existing DB and rebuilds
"""

import json
import re
import argparse
import time
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
LAW_FILES = [
    "data/constitution_final.json",
    "data/bns_final.json",
    "data/bnss_final.json",
]
CHROMA_DIR   = "./chroma_db"
COLLECTION   = "constitution_of_india"
EMBED_MODEL  = "BAAI/bge-small-en-v1.5"

# Chunk size thresholds (in characters)
SMALL_ENTRY_MAX  = 1500   # entries ≤ this are kept as a single chunk
LARGE_CHUNK_SIZE = 1000   # target max chars per chunk when splitting large entries

# ─────────────────────────────────────────────
# STEP 1: CHUNKING
# ─────────────────────────────────────────────

def build_context_prefix(entry: dict) -> str:
    """
    Build a human-readable header that gets prepended to every chunk.
    This steers the embedding model toward the right semantic region.

    Example output:
      "Article 21 — Protection of life and personal liberty
       (Part III: Fundamental Rights | Type: article)"
    """
    parts = []

    entry_type = entry.get("type", "article")

    if entry_type == "preamble":
        parts.append("Preamble of the Constitution of India")

    elif entry_type == "article":
        article_num = entry.get("article", "")
        title       = entry.get("title", "")
        part        = entry.get("part", "")
        part_name   = entry.get("part_name", "")
        topic       = entry.get("topic", "")

        line1 = f"Article {article_num}"
        if title:
            line1 += f" — {title}"
        parts.append(line1)

        context_bits = []
        if part and part_name:
            context_bits.append(f"Part {part}: {part_name}")
        if topic and topic != part_name:
            context_bits.append(f"Topic: {topic}")
        if context_bits:
            parts.append(f"({' | '.join(context_bits)})")

    elif entry_type == "schedule":
        parts.append(f"{entry.get('title', 'Schedule')} — {entry.get('topic', '')}")

    elif entry_type == "amendment":
        parts.append(f"{entry.get('article', 'Amendment')} — {entry.get('title', '')}")

    elif entry_type == "section":
        section_num = entry.get("section", "")
        title       = entry.get("title", "")
        chapter     = entry.get("chapter", "")
        chapter_name= entry.get("chapter_name", "")
        topic       = entry.get("topic", "")
        law         = entry.get("law", "BNS")
        ipc         = entry.get("ipc_equivalent", "")
        crpc        = entry.get("crpc_equivalent", "")

        line1 = f"{law} - Section {section_num}"
        if title:
            line1 += f" — {title}"
        if ipc:
            line1 += f" [Old IPC: {ipc}]"
        if crpc:
            line1 += f" [Old CrPC: {crpc}]"
        parts.append(line1)

        context_bits = []
        if chapter and chapter_name:
            context_bits.append(f"Chapter {chapter}: {chapter_name}")
        if topic and topic != chapter_name:
            context_bits.append(f"Topic: {topic}")
        if context_bits:
            parts.append(f"({' | '.join(context_bits)})")

    # Add status warning for non-active entries
    status = entry.get("status", "active")
    if status != "active":
        parts.append(f"[STATUS: {status}]")

    return "\n".join(parts)


def split_by_clauses(text: str, max_size: int = LARGE_CHUNK_SIZE) -> list[str]:
    """
    Split a long article description into clause-aware chunks.
    
    Strategy:
      1. Try to split on numbered clauses: (1), (2), (3)... or (a), (b)...
      2. Fall back to sentence-boundary splitting if clauses are still too long
      3. Never cut mid-word — always split on whitespace/punctuation
    
    Returns a list of text pieces (no context prefix yet).
    """
    if len(text) <= max_size:
        return [text.strip()]

    # Try splitting on top-level numbered clauses like "(1) ...", "(2) ..."
    # These are the primary structural dividers in Indian constitutional text
    clause_pattern = re.compile(r'(?=\(\d+\)\s)')
    clause_splits = clause_pattern.split(text)
    clause_splits = [c.strip() for c in clause_splits if c.strip()]

    # If splitting gave us reasonably-sized pieces, use them
    if len(clause_splits) > 1:
        chunks = []
        current = ""
        for clause in clause_splits:
            if len(current) + len(clause) + 2 <= max_size:
                current = (current + "\n\n" + clause).strip()
            else:
                if current:
                    chunks.append(current)
                # If single clause is too long, split it further by sentences
                if len(clause) > max_size:
                    chunks.extend(split_by_sentences(clause, max_size))
                else:
                    current = clause
        if current:
            chunks.append(current)
        return chunks

    # No clause structure found — split by sentences
    return split_by_sentences(text, max_size)


def split_by_sentences(text: str, max_size: int = LARGE_CHUNK_SIZE) -> list[str]:
    """
    Fallback splitter: split on sentence boundaries (. or ;\n or :\n).
    Always tries to keep sentences together within max_size.
    """
    # Split on sentence-ending punctuation followed by whitespace/newline
    sentences = re.split(r'(?<=[.;:])\s+', text)
    
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 <= max_size:
            current = (current + " " + sent).strip()
        else:
            if current:
                chunks.append(current)
            # If a single sentence is over max_size, hard-split on whitespace
            if len(sent) > max_size:
                words = sent.split()
                line = ""
                for word in words:
                    if len(line) + len(word) + 1 <= max_size:
                        line = (line + " " + word).strip()
                    else:
                        if line:
                            chunks.append(line)
                        line = word
                if line:
                    chunks.append(line)
            else:
                current = sent
    if current:
        chunks.append(current)
    
    return chunks if chunks else [text.strip()]


def chunk_entry(entry: dict) -> list[dict]:
    """
    Convert a single JSON entry into one or more chunks with metadata.
    
    Returns a list of dicts:
      {
        "chunk_id": str,       # unique id like "article_21_chunk_0"
        "text":     str,       # context_prefix + "\n\n" + description_piece
        "metadata": dict       # all fields needed for citation + filtering
      }
    """
    entry_id    = entry.get("id", "unknown")
    description = entry.get("description", "").strip()
    entry_type  = entry.get("type", "article")
    status      = entry.get("status", "active")

    if not description:
        return []

    context_prefix = build_context_prefix(entry)

    # Decide how to split based on type and length
    if entry_type == "preamble":
        # Preamble is short, always single chunk
        pieces = [description]

    elif entry_type == "amendment":
        # Amendments are 122-632 chars — always single chunk
        pieces = [description]

    elif entry_type == "schedule":
        # Schedules can be huge (62k chars). Split aggressively.
        pieces = split_by_clauses(description, max_size=LARGE_CHUNK_SIZE)

    elif entry_type in ["article", "section"]:
        if len(description) <= SMALL_ENTRY_MAX:
            # Short article/section — single chunk, no splitting
            pieces = [description]
        else:
            # Long article/section — clause-aware splitting
            pieces = split_by_clauses(description, max_size=LARGE_CHUNK_SIZE)
    else:
        pieces = [description]

    # Build chunks with full context
    chunks = []
    for i, piece in enumerate(pieces):
        # The text field = context header + actual content
        # The header helps the embedding model understand context
        full_text = f"{context_prefix}\n\n{piece}"

        # Metadata — stored alongside the vector in ChromaDB
        # These fields are used for: citations, filtering, debug
        metadata = {
            "source_id":    entry_id,
            "article":      str(entry.get("article", "")),
            "section":      str(entry.get("section", "")),
            "title":        str(entry.get("title", "")),
            "part":         str(entry.get("part", "") or ""),
            "part_name":    str(entry.get("part_name", "") or ""),
            "chapter":      str(entry.get("chapter", "") or ""),
            "chapter_name": str(entry.get("chapter_name", "") or ""),
            "topic":        str(entry.get("topic", "") or ""),
            "type":         entry_type,
            "law":          str(entry.get("law", "Constitution")),
            "ipc_equivalent": str(entry.get("ipc_equivalent", "") or ""),
            "crpc_equivalent": str(entry.get("crpc_equivalent", "") or ""),
            "status":       str(status),
            "chunk_index":  i,
            "total_chunks": len(pieces),
            "is_active":    str(status == "active"),  # ChromaDB metadata must be str/int/float
        }

        chunk_id = f"{entry_id}_chunk_{i}"

        chunks.append({
            "chunk_id": chunk_id,
            "text":     full_text,
            "metadata": metadata,
        })

    return chunks


def chunk_all_entries(entries: list[dict]) -> list[dict]:
    """
    Chunk all 585 entries. Returns flat list of all chunks.
    Prints a summary of what was created.
    """
    all_chunks = []
    skipped    = 0

    for entry in entries:
        result = chunk_entry(entry)
        if not result:
            skipped += 1
        all_chunks.extend(result)

    # Stats
    type_counts = {}
    for c in all_chunks:
        t = c["metadata"]["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"\n{'─'*50}")
    print(f"  CHUNKING COMPLETE")
    print(f"{'─'*50}")
    print(f"  Input entries  : {len(entries)}")
    print(f"  Output chunks  : {len(all_chunks)}")
    print(f"  Skipped (empty): {skipped}")
    print(f"  Breakdown by type:")
    for t, n in sorted(type_counts.items()):
        print(f"    {t:<12}: {n} chunks")

    # Show chunk size distribution
    sizes = [len(c["text"]) for c in all_chunks]
    print(f"\n  Chunk size stats (characters):")
    print(f"    Min  : {min(sizes)}")
    print(f"    Max  : {max(sizes)}")
    print(f"    Avg  : {int(sum(sizes)/len(sizes))}")
    print(f"    >1500: {sum(1 for s in sizes if s > 1500)} chunks")
    print(f"{'─'*50}\n")

    return all_chunks


# ─────────────────────────────────────────────
# STEP 2+3: EMBED & STORE (ChromaDB handles both)
# ─────────────────────────────────────────────

def build_vector_store(chunks: list[dict], reset: bool = False):
    """
    Store all chunks into ChromaDB.
    ChromaDB calls the embedding model internally — you don't manage vectors manually.
    
    Args:
        chunks: Output from chunk_all_entries()
        reset:  If True, delete existing collection and rebuild from scratch
    """
    print(f"  Initialising ChromaDB at: {CHROMA_DIR}")
    print(f"  Embedding model: {EMBED_MODEL}")
    print(f"  (First run downloads ~130MB model from HuggingFace)\n")

    # Point ChromaDB to use bge-small-en-v1.5 for all embeddings
    # This same model MUST be used at query time too — consistency is critical
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Handle reset
    if reset:
        try:
            client.delete_collection(COLLECTION)
            print(f"  Deleted existing collection: {COLLECTION}")
        except Exception:
            pass

    # Get or create collection
    collection = client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},  # Use cosine similarity (standard for text)
    )

    existing_count = collection.count()
    if existing_count > 0 and not reset:
        print(f"  Collection already has {existing_count} chunks.")
        print(f"  Run with --reset to rebuild from scratch.\n")
        return collection

    # Batch insert — ChromaDB is fast with batches of 100-500
    BATCH_SIZE = 200
    total = len(chunks)

    print(f"  Embedding + storing {total} chunks in batches of {BATCH_SIZE}...")
    start_time = time.time()

    for batch_start in range(0, total, BATCH_SIZE):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]

        collection.add(
            ids       = [c["chunk_id"]  for c in batch],
            documents = [c["text"]      for c in batch],
            metadatas = [c["metadata"]  for c in batch],
        )

        done = min(batch_start + BATCH_SIZE, total)
        elapsed = time.time() - start_time
        eta = (elapsed / done) * (total - done) if done > 0 else 0
        print(f"  [{done:>4}/{total}] stored  |  {elapsed:.1f}s elapsed  |  ETA: {eta:.1f}s")

    elapsed = time.time() - start_time
    print(f"\n{'─'*50}")
    print(f"  EMBEDDING + STORAGE COMPLETE")
    print(f"{'─'*50}")
    print(f"  Total chunks stored : {collection.count()}")
    print(f"  Time taken          : {elapsed:.1f} seconds")
    print(f"  ChromaDB location   : {CHROMA_DIR}/")
    print(f"{'─'*50}\n")

    return collection


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LawBook India — RAG Ingestion Script")
    parser.add_argument("--reset", action="store_true", help="Delete and rebuild the vector store")
    args = parser.parse_args()

    all_entries = []
    
    # ── Load ──
    for fpath in LAW_FILES:
        json_path = Path(fpath)
        if not json_path.exists():
            print(f"WARNING: JSON file not found at '{json_path}'. Skipping.")
            continue
            
        print(f"\nLoading data from: {json_path}")
        with open(json_path, encoding="utf-8") as f:
            entries = json.load(f)
            all_entries.extend(entries)
        print(f"  Loaded {len(entries)} entries\n")

    if not all_entries:
        print("ERROR: No entries loaded. Make sure the JSON files exist.")
        return

    # ── Step 2.1: Chunk ──
    print("STEP 1/2 — Chunking entries...")
    chunks = chunk_all_entries(all_entries)

    # ── Steps 2.2 + 2.3: Embed + Store ──
    print("STEP 2/2 — Embedding + storing in ChromaDB...")
    build_vector_store(chunks, reset=args.reset)

    print("Done! You can now run: python test_retrieval.py")


if __name__ == "__main__":
    main()