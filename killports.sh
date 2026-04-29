#!/bin/bash
# Utility to kill processes occupying ports 5173-5180 and 8000

echo "Killing processes on common editor ports..."

# Kill processes on ports 5173-5180 (frontend)
for port in 5173 5174 5175 5176 5177 5178 5179 5180; do
    if lsof -i :$port >/dev/null 2>&1; then
        echo "Killing process on port $port..."
        lsof -ti :$port | xargs kill -9 2>/dev/null || true
    fi
done

# Kill process on port 8000 (backend)
if lsof -i :8000 >/dev/null 2>&1; then
    echo "Killing process on port 8000..."
    lsof -ti :8000 | xargs kill -9 2>/dev/null || true
fi

# Also kill any lingering npm/vite/python processes
pkill -f "npm run dev" 2>/dev/null || true
pkill -f "vite" 2>/dev/null || true
pkill -f "python server.py" 2>/dev/null || true

echo "✓ Cleanup complete. Ports 5173-5180 and 8000 are now free."
