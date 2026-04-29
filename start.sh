#!/bin/bash
# Start script for Linux/Mac
# Starts both backend and frontend servers

set -e

echo ""
echo "============================================"
echo "Starting Editor Backend and Frontend"
echo "============================================"
echo ""

# Load environment
if [ ! -f .env ]; then
    echo "ERROR: .env file not found"
    echo "Please run ./setup.sh first"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

echo "[1/2] Starting FastAPI backend on http://127.0.0.1:8000"
echo "      (Backend will be running in this terminal)"
echo ""
echo "[2/2] Frontend will be available at http://127.0.0.1:5173"
echo "      (Frontend will start in a new terminal)"
echo ""
echo "Press Ctrl+C to stop the backend when done"
echo ""

# Start frontend in background
(cd frontend && npm run dev) &
FRONTEND_PID=$!

# Give frontend a moment to start
sleep 2

# Start backend (foreground)
python server.py

# Cleanup frontend on exit
kill $FRONTEND_PID 2>/dev/null || true
