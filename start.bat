@echo off
echo Starting AI PDF Assistant Full-Stack Application...
echo.

start "AI PDF Assistant - FastAPI Backend" cmd /k ".venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload"

echo Waiting for backend to start...
timeout /t 3 /nobreak >nul

start "AI PDF Assistant - React Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo Application started!
echo Frontend: http://localhost:5173
echo Backend API: http://127.0.0.1:8000/docs
echo.
pause
