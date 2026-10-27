@echo off
echo ========================================================
echo   Starting PhytoVision AI - Full-Stack Diagnostic Engine
echo ========================================================
echo.

REM 1. Activate Virtual Environment and Launch FastAPI Backend in a separate window
echo [1/2] Starting Backend API Server (Port 8000)...
start "PhytoVision AI Backend" cmd /k ".\.venv\Scripts\python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload"

REM 2. Start Vite Frontend Dev Server
echo [2/2] Starting Frontend Dev Server (Port 5173)...
cd frontend
npm run dev