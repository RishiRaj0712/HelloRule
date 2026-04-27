#!/bin/bash
echo "Building vector store..."
python3 ingest.py --json data/constitution_final.json
echo "Starting FastAPI server..."
uvicorn main:app --host 0.0.0.0 --port $PORT
