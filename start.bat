@echo off
cd /d "%~dp0"
echo ========================================
echo    Hamro Mitra AI Assistant
echo    Created by Anup Raj Uprety
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Install dependencies if needed
echo 📦 Installing/updating dependencies...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

echo ✅ Dependencies installed
echo.

REM Start the unified backend
echo 🚀 Starting Unified AI Backend...
start /B python unified_ai_backend.py

REM Wait for backend to start
echo ⏳ Waiting for backend to initialize...
timeout /t 3 /nobreak >nul

REM Open the frontend
echo 🌐 Opening AI Assistant in browser...
start index.html

echo.
echo ========================================
echo 🎉 Hamro Mitra is now running!
echo ========================================
echo.
echo 📊 System Status:
echo    • Backend: http://localhost:5000
echo    • Frontend: Opened in browser
echo    • AI Sources: ChatGPT, Gemini, Web Search
echo.
echo 💡 Features:
echo    • Real-time web search
echo    • Multiple AI model integration
echo    • Intelligent query routing
echo    • Current events and factual queries
echo.
echo 🛑 To stop: Close this window
echo.

pause
