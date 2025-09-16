@echo off
echo ========================================
echo   MEDICAL CT ANALYSIS WITH AI AGENTS
echo   Real Implementation with Your Model
echo ========================================

echo.
echo Starting backend with your trained model...
start /B "Backend" cmd /k "cd /D C:\temp\app\backend && python app.py"

echo Waiting for backend to start...
timeout /t 5 >nul

echo.
echo Starting frontend...
start /B "Frontend" cmd /k "cd /D C:\temp\app\frontend && npm install && npm run dev"

echo.
echo ========================================
echo   APPLICATION STARTING!
echo ========================================
echo   Backend:  http://localhost:5000
echo   Frontend: http://localhost:3000
echo ========================================
echo.
echo Press any key to view logs or Ctrl+C to stop...
pause >nul
