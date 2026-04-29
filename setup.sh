#!/bin/bash
# Setup script for Linux/Mac
# This script sets up the project for first-time use

set -e  # Exit on error

echo ""
echo "============================================"
echo "Project Setup - Linux/Mac"
echo "============================================"
echo ""

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.10+ from https://www.python.org"
    exit 1
fi

# Check Node.js installation
if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js is not installed or not in PATH"
    echo "Please install Node.js from https://nodejs.org"
    exit 1
fi

echo "[1/4] Creating Python virtual environment..."
python3 -m venv venv

echo "[2/4] Activating virtual environment and installing dependencies..."
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo "[3/4] Setting up environment file..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file from .env.example"
    echo "NOTE: Update .env with your configuration if needed"
else
    echo ".env already exists, skipping"
fi

echo "[4/4] Installing frontend dependencies..."
cd frontend
npm install
cd ..

echo ""
echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Review and update .env if needed"
echo "  2. Run: ./start.sh"
echo ""
