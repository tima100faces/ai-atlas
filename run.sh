#!/usr/bin/env bash
# AI Atlas — Local development runner
# Starts both backend and frontend.

set -e

DIR="$(cd "$(dirname "$0")" && pwd)"

echo "==> Starting backend (FastAPI on :8001)..."
cd "$DIR/app/backend"
source venv/bin/activate 2>/dev/null || (python3 -m venv venv && source venv/bin/activate && pip install -q -r requirements.txt)
python3 -m uvicorn main:app --host 0.0.0.0 --port 8001 &
BACKEND_PID=$!

echo "==> Starting frontend (Vite on :5173)..."
cd "$DIR/app/frontend"
npm run dev -- --host 0.0.0.0 &
FRONTEND_PID=$!

echo ""
echo "  Backend:  http://localhost:8001"
echo "  Frontend: http://localhost:5173"
echo "  Password: timohin2026"
echo ""
echo "Press Ctrl+C to stop."

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait
