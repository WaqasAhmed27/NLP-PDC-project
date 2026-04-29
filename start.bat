@echo off
REM Start script for Windows
REM Starts both backend and frontend servers

echo.
echo ============================================
echo Starting Editor Backend and Frontend
echo ============================================
echo.

REM Load environment
if not exist .env (
    echo ERROR: .env file not found
    echo Please run setup.bat first
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if venv was activated
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    echo Please run setup.bat first
    pause
    exit /b 1
)

echo [1/2] Starting FastAPI backend on http://127.0.0.1:8000
echo       (Backend will be running in this terminal)
echo.
echo [2/2] Frontend will be available at http://127.0.0.1:5173
echo       (Frontend will start in a new terminal)
echo.
echo Press Ctrl+C to stop the backend when done
echo.

REM Start frontend in a new terminal
start "Frontend Dev Server" cmd /k "cd frontend && npm run dev"

REM Give frontend a moment to start
timeout /t 2 /nobreak

REM Start backend
python server.py

REM Keep window open if there was an error
if errorlevel 1 (
    echo.
    echo Backend exited with an error
    pause
)
