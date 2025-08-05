@echo off
echo 🚀 Starting AI FallGuard Frontend Demo...
echo ==========================================

REM Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js is not installed. Please install Node.js first.
    pause
    exit /b 1
)

REM Check if npm is installed
npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ npm is not installed. Please install npm first.
    pause
    exit /b 1
)

REM Check if dependencies are installed
if not exist "node_modules" (
    echo 📦 Installing dependencies...
    npm install
)

REM Start the development server
echo 🌐 Starting development server...
echo 📍 Demo will be available at: http://localhost:3000
echo 📱 Open your browser and navigate to the URL above
echo 🎥 Click 'Start Camera' to begin the live demo
echo.
echo Press Ctrl+C to stop the server
echo.

npm run dev 