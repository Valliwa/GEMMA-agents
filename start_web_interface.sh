#!/bin/bash

echo "🌐 Starting SICA Web Interface..."
echo "=================================="

# Check if Gemma 3 server is running
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Gemma 3 server detected on localhost:8000"
else
    echo "⚠️ Gemma 3 server not detected on localhost:8000"
    echo "   Make sure gemma_api_server.py is running"
fi

echo ""
echo "📍 Starting web interface on http://localhost:5000"
echo "🔧 Features available:"
echo "   - Real-time Gemma 3 monitoring"
echo "   - Interactive agent testing"
echo "   - SICA experiment analysis"
echo "   - Benchmark trace examination"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 sica_web_interface.py
