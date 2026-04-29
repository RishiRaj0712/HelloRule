#!/bin/bash
echo "Building vector store..."
python3 ingest.py
echo "Starting FastAPI server..."
uvicorn main:app --host 0.0.0.0 --port $PORT
