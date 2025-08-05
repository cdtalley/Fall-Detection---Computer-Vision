#!/bin/bash

echo "🚀 Starting AI FallGuard Frontend Demo..."
echo "=========================================="

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js first."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm first."
    exit 1
fi

# Check if dependencies are installed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Start the development server
echo "🌐 Starting development server..."
echo "📍 Demo will be available at: http://localhost:3000"
echo "📱 Open your browser and navigate to the URL above"
echo "🎥 Click 'Start Camera' to begin the live demo"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

npm run dev 