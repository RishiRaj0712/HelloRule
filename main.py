"""
main.py — LawBook India Backend
=================================
FastAPI application. Wires retriever → prompt → generator into
a single /api/chat endpoint that the React frontend will call.

Endpoints:
  GET  /health          → health check
  POST /api/chat        → main Q&A endpoint (full response)
  POST /api/chat/stream → streaming Q&A endpoint (token by token)

Run with:
  uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json

from retriever import Retriever
from generator import Generator
from prompt import build_prompt, extract_sources_from_chunks


# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────

app = FastAPI(
    title="LawBook India API",
    description="RAG-based Q&A on the Constitution of India",
    version="1.0.0",
)

# CORS — allows the React frontend (localhost:3000) to call this API
# In production, replace "*" with your actual frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Change to ["https://yourdomain.com"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# STARTUP — load models once
# ─────────────────────────────────────────────
# These are loaded when the server starts, not on every request.
# Loading the embedding model takes ~3 seconds — we don't want
# that delay on every user query.

retriever = None
generator = None

@app.on_event("startup")
async def startup_event():
    global retriever, generator
    print("\n[LawBook] Starting up...")
    retriever = Retriever()   # loads ChromaDB + embedding model
    generator = Generator()   # configures Gemini API
    print("[LawBook] Ready to serve requests\n")


# ─────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    history: list[dict] = []   # [{"role": "user"|"assistant", "content": "..."}]
    top_k: int = 5             # how many chunks to retrieve (default 5)

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Can police arrest someone without a warrant?",
                "history": [],
                "top_k": 5
            }
        }


class SourceItem(BaseModel):
    article:   str
    title:     str
    part:      str
    part_name: str
    type:      str
    status:    str
    score:     float


class ChatResponse(BaseModel):
    answer:     str
    sources:    list[SourceItem]
    model_used: str
    chunks_used: int


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """
    Quick check to verify the server is running and models are loaded.
    Call this before starting the frontend to confirm the backend is ready.
    """
    return {
        "status":     "ok",
        "retriever":  retriever is not None,
        "generator":  generator is not None,
        "collection": "constitution_of_india",
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main Q&A endpoint.

    Flow:
      1. Validate query
      2. Retrieve top-K relevant chunks from ChromaDB
      3. Build structured prompt (system + context + history + question)
      4. Send to Gemini, get full answer
      5. Return answer + sources + metadata

    The full response is returned at once (~2-5 seconds).
    Use /api/chat/stream for a streaming typing-effect response.
    """
    if not retriever or not generator:
        raise HTTPException(status_code=503, detail="Models not yet loaded. Try again in a moment.")

    # 1. Validate
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if len(query) > 1000:
        raise HTTPException(status_code=400, detail="Query too long (max 1000 characters).")

    # 2. Retrieve
    chunks = retriever.search_with_fallback(query, top_k=req.top_k)

    # 3. Build prompt
    prompt = build_prompt(query, chunks, req.history)

    # 4. Generate
    answer = generator.generate(prompt)

    # 5. Extract sources for frontend display
    sources = extract_sources_from_chunks(chunks)

    return ChatResponse(
        answer=answer,
        sources=sources,
        model_used="gemini-2.5-flash",
        chunks_used=len(chunks),
    )


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Streaming Q&A endpoint using Server-Sent Events (SSE).

    Same as /api/chat but streams the answer token by token.
    The frontend receives chunks of text as they're generated,
    enabling the "typing" effect.

    Response format: text/event-stream
    Each event: data: {"type": "token"|"sources"|"done", ...}
    """
    if not retriever or not generator:
        raise HTTPException(status_code=503, detail="Models not yet loaded.")

    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Retrieve and build prompt (same as non-streaming)
    chunks  = retriever.search_with_fallback(query, top_k=req.top_k)
    prompt  = build_prompt(query, chunks, req.history)
    sources = extract_sources_from_chunks(chunks)

    def event_stream():
        """
        Generator that yields SSE-formatted events.

        Event types:
          token   → a piece of the answer text
          sources → the source articles used (sent after answer completes)
          done    → signals the stream is finished
        """
        # Stream answer tokens
        for token in generator.generate_stream(prompt):
            data = json.dumps({"type": "token", "content": token})
            yield f"data: {data}\n\n"

        # Send sources after answer is complete
        sources_data = json.dumps({
            "type":    "sources",
            "sources": [s.model_dump() if hasattr(s, 'model_dump') else s for s in sources],
        })
        yield f"data: {sources_data}\n\n"

        # Signal completion
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disables Nginx buffering in production
        },
    )


# ─────────────────────────────────────────────
# DEV ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)