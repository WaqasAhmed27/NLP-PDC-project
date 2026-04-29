@echo off
REM Setup script for Windows
REM This script sets up the project for first-time use

echo.
echo ============================================
echo Project Setup - Windows
echo ============================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://www.python.org
    pause
    exit /b 1
)

REM Check Node.js installation
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org
    pause
    exit /b 1
)

echo [1/4] Creating Python virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo [2/4] Activating virtual environment and installing dependencies...
call venv\Scripts\activate.bat
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install Python dependencies
    pause
    exit /b 1
)

echo [3/4] Setting up environment file...
if not exist .env (
    copy .env.example .env
    echo Created .env file from .env.example
    echo NOTE: Update .env with your configuration if needed
) else (
    echo .env already exists, skipping
)

echo [4/4] Installing frontend dependencies...
cd frontend
call npm install
if errorlevel 1 (
    echo ERROR: Failed to install frontend dependencies
    cd ..
    pause
    exit /b 1
)
cd ..

echo.
echo ============================================
echo Setup Complete!
echo ============================================
echo.
echo Next steps:
echo   1. Review and update .env if needed
echo   2. Run: start.bat
echo.
pause
