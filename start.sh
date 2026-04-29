#!/bin/bash
# Start script for Linux/Mac
# Starts both backend and frontend servers

set -e

# Cleanup function for graceful shutdown
cleanup() {
    echo ""
    echo "Shutting down..."
    
    # Kill frontend process group
    if [ ! -z "$FRONTEND_PID" ]; then
        kill -TERM $FRONTEND_PID 2>/dev/null || true
        wait $FRONTEND_PID 2>/dev/null || true
    fi
    
    # Kill any remaining npm/vite processes
    pkill -f "npm run dev" 2>/dev/null || true
    pkill -f "vite" 2>/dev/null || true
    
    echo "✓ Cleanup complete"
    exit 0
}

# Register cleanup on Ctrl+C and other signals
trap cleanup SIGINT SIGTERM EXIT

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

# Unset any previously set environment variables
unset EXLLAMA_MODEL_DIR
unset USE_MOCK_ENGINE

# Set CUDA_HOME for ExLlamaV2
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=/usr/local/cuda-12.4/bin:$PATH

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

# Cleanup on normal exit
cleanup
