#!/bin/bash

# start_sica_web.sh - One-click SICA Web Interface launcher

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

print_banner() {
    echo -e "${BLUE}"
    echo "  ██████  ██  ██████ █████      ██     ██ ███████ ██████  "
    echo " ██       ██ ██      ██   ██    ██     ██ ██      ██   ██ "
    echo " ██████   ██ ██      ███████    ██  █  ██ █████   ██████  "
    echo "      ██  ██ ██      ██   ██    ██ ███ ██ ██      ██   ██ "
    echo " ██████   ██  ██████ ██   ██     ███ ███  ███████ ██████  "
    echo -e "${NC}"
    echo -e "${PURPLE}Self-Improving Coding Agent - Web Interface${NC}"
    echo -e "${BLUE}=============================================${NC}"
}

check_gemma3_server() {
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Gemma 3 server detected and running${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️ Gemma 3 server not detected${NC}"
        return 1
    fi
}

check_dependencies() {
    echo "🔍 Checking dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python 3 not found${NC}"
        return 1
    fi
    
    # Check Flask
    if ! python3 -c "import flask" 2>/dev/null; then
        echo -e "${YELLOW}⚠️ Flask not installed${NC}"
        echo "Installing Flask and dependencies..."
        pip install flask flask-socketio requests
        if [ $? -ne 0 ]; then
            echo -e "${RED}❌ Failed to install dependencies${NC}"
            return 1
        fi
    fi
    
    echo -e "${GREEN}✅ Dependencies OK${NC}"
    return 0
}

setup_files() {
    # Check if web interface files exist
    if [ ! -f "sica_web_interface.py" ]; then
        echo -e "${YELLOW}⚠️ Web interface files not found${NC}"
        echo "Please create the web interface files first:"
        echo "1. sica_web_interface.py"
        echo "2. templates/ directory with HTML files"
        echo ""
        echo "Run the setup script: ./setup_web_interface.sh"
        return 1
    fi
    
    # Create templates directory if it doesn't exist
    if [ ! -d "templates" ]; then
        echo "📁 Creating templates directory..."
        mkdir -p templates static
    fi
    
    return 0
}

main() {
    print_banner
    
    echo "🚀 Starting SICA Web Interface..."
    echo ""
    
    # Check dependencies
    if ! check_dependencies; then
        echo ""
        echo -e "${RED}💡 Setup required. Please run:${NC}"
        echo "   pip install flask flask-socketio requests"
        exit 1
    fi
    
    # Check if files are set up
    if ! setup_files; then
        echo ""
        echo -e "${RED}💡 Setup required. Please create the web interface files first.${NC}"
        exit 1
    fi
    
    # Check Gemma 3 server
    echo ""
    if ! check_gemma3_server; then
        echo ""
        echo -e "${YELLOW}💡 To get the most out of the web interface:${NC}"
        echo "   1. Start your Gemma 3 server: python3 gemma_api_server.py"
        echo "   2. Wait for it to load completely"
        echo "   3. Then restart this web interface"
        echo ""
        echo -e "${BLUE}Continuing anyway...${NC}"
    fi
    
    echo ""
    echo -e "${GREEN}🌐 Starting SICA Web Interface...${NC}"
    echo ""
    echo -e "${PURPLE}📍 Web interface will be available at:${NC}"
    echo "   http://localhost:8080"
    echo ""
    echo -e "${PURPLE}🎯 Features available:${NC}"
    echo "   ✅ Real-time Gemma 3 monitoring"
    echo "   ✅ Interactive agent testing"
    echo "   ✅ SICA experiment analysis"
    echo "   ✅ Benchmark trace examination"
    echo "   ✅ GPU performance monitoring"
    echo ""
    echo -e "${BLUE}📊 Monitoring:${NC}"
    echo "   - Gemma 3 server: http://localhost:8000"
    echo "   - SICA experiments: ./results/"
    echo "   - Web interface: http://localhost:8080"
    echo ""
    echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
    echo ""
    echo "Starting in 3 seconds..."
    sleep 3
    
    # Start the web interface
    if [ -f "sica_web_interface.py" ]; then
        python3 sica_web_interface.py
    else
        echo -e "${RED}❌ sica_web_interface.py not found!${NC}"
        echo ""
        echo "Please make sure you have created the web interface files."
        echo "You may need to copy the code from the provided artifacts."
        exit 1
    fi
}

# Handle command line arguments
case "$1" in
    --help|-h)
        echo "SICA Web Interface Launcher"
        echo ""
        echo "Usage: $0 [OPTION]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --check        Check system requirements only"
        echo "  --setup        Run setup assistance"
        echo ""
        echo "This script launches the SICA web interface with automatic checks."
        exit 0
        ;;
    --check)
        print_banner
        echo "🔍 System Check"
        echo ""
        check_dependencies
        check_gemma3_server
        echo ""
        echo "Check complete."
        exit 0
        ;;
    --setup)
        print_banner
        echo "🛠️ Setup Assistance"
        echo ""
        echo "To set up the SICA web interface:"
        echo ""
        echo "1. Install dependencies:"
        echo "   pip install flask flask-socketio requests"
        echo ""
        echo "2. Create the web interface files:"
        echo "   - Copy sica_web_interface.py from the provided code"
        echo "   - Create templates/ directory with HTML files"
        echo "   - Run: ./setup_web_interface.sh (if available)"
        echo ""
        echo "3. Start your Gemma 3 server:"
        echo "   python3 gemma_api_server.py"
        echo ""
        echo "4. Launch the web interface:"
        echo "   ./start_sica_web.sh"
        echo ""
        exit 0
        ;;
    "")
        main
        ;;
    *)
        echo "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac
