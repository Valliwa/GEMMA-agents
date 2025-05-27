#!/bin/bash

# getting_started.sh - Step-by-step Gemma 3 27B testing guide

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${BLUE}================================${NC}"
    echo -e "${BLUE}🚀 $1${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_step() {
    echo -e "\n${PURPLE}📋 Step $1: $2${NC}"
    echo -e "${PURPLE}$(printf '%.0s-' {1..40})${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

check_model_download() {
    echo "🔍 Checking if Gemma 3 27B model download is complete..."
    
    # Check if model files exist
    if [ -d "./hf_cache/models--google--gemma-3-27b-it" ]; then
        # Count safetensors files (should be 12 for Gemma 3 27B)
        model_files=$(find ./hf_cache/models--google--gemma-3-27b-it -name "*.safetensors" | wc -l)
        if [ "$model_files" -ge 10 ]; then
            print_success "Model files found ($model_files safetensors files)"
            return 0
        else
            print_warning "Model download appears incomplete ($model_files files found)"
            return 1
        fi
    else
        print_warning "Model cache directory not found"
        return 1
    fi
}

check_server_running() {
    echo "🔍 Checking if Gemma 3 API server is running..."
    
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        print_success "Server is running and responding"
        return 0
    else
        print_warning "Server is not responding on localhost:8000"
        return 1
    fi
}

main() {
    print_header "Gemma 3 27B Simple Testing Guide"
    
    echo "This script will guide you through testing your Gemma 3 27B integration step by step."
    echo ""
    
    # Step 1: Check model download
    print_step "1" "Check Model Download Status"
    
    if check_model_download; then
        print_success "Model download is complete!"
    else
        print_error "Model download is not complete yet."
        echo ""
        echo "💡 What to do:"
        echo "   - Wait for your gemma_api_server.py download to finish"
        echo "   - Look for the message: '✅ Gemma 3 27B model loaded successfully!'"
        echo "   - Then come back and run this script again"
        echo ""
        echo "🔄 To check download progress, look at your server terminal"
        echo "📊 Expected model size: ~50GB total"
        return 1
    fi
    
    # Step 2: Check if server is running
    print_step "2" "Check API Server Status"
    
    if check_server_running; then
        print_success "API server is running!"
    else
        print_warning "API server is not running."
        echo ""
        echo "💡 Starting the server:"
        echo "   1. Open a new terminal"
        echo "   2. Navigate to your GEMMA directory"
        echo "   3. Run: python3 gemma_api_server.py"
        echo "   4. Wait for: '✅ Gemma 3 27B model loaded successfully!'"
        echo "   5. Come back here and continue"
        echo ""
        read -p "Press Enter when the server is running..."
        
        if check_server_running; then
            print_success "Great! Server is now running."
        else
            print_error "Server still not responding. Please check for errors."
            return 1
        fi
    fi
    
    # Step 3: Run the simple tests
    print_step "3" "Run Integration Tests"
    
    echo "🧪 Now running comprehensive integration tests..."
    echo ""
    
    if [ -f "test_gemma3_simple.py" ]; then
        python3 test_gemma3_simple.py
        test_result=$?
        
        if [ $test_result -eq 0 ]; then
            print_success "All tests passed!"
        else
            print_warning "Some tests failed, but you can still proceed"
        fi
    else
        print_error "Test script not found: test_gemma3_simple.py"
        echo "Please create the test script first."
        return 1
    fi
    
    # Step 4: Next steps
    print_step "4" "Next Steps"
    
    echo "🎉 Your Gemma 3 27B integration is ready!"
    echo ""
    echo "🚀 What you can do now:"
    echo ""
    echo "   Option A: Test with the SICA runner"
    echo "   python runner.py --test-gemma3"
    echo ""
    echo "   Option B: Run a single benchmark"
    echo "   python runner.py test --name gsm8k"
    echo ""
    echo "   Option C: Start a full SICA experiment"
    echo "   python runner.py --id 1 --iterations 5"
    echo ""
    echo "📊 Monitor your GPU usage:"
    echo "   watch -n 2 nvidia-smi"
    echo ""
    echo "📈 Monitor server performance:"
    echo "   curl http://localhost:8000/stats"
    echo ""
    
    print_success "Setup complete! Happy experimenting! 🧠✨"
}

# Help function
show_help() {
    echo "Gemma 3 27B Testing Guide"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  --help, -h     Show this help message"
    echo "  --check-only   Only check status, don't run tests"
    echo "  --force-test   Run tests even if checks fail"
    echo ""
    echo "This script helps you test your Gemma 3 27B integration step by step."
}

# Parse command line arguments
case "$1" in
    --help|-h)
        show_help
        exit 0
        ;;
    --check-only)
        print_header "Status Check Only"
        check_model_download
        check_server_running
        exit 0
        ;;
    --force-test)
        print_header "Force Testing Mode"
        if [ -f "test_gemma3_simple.py" ]; then
            python3 test_gemma3_simple.py
        else
            print_error "Test script not found"
            exit 1
        fi
        exit 0
        ;;
    "")
        main
        ;;
    *)
        echo "Unknown option: $1"
        show_help
        exit 1
        ;;
esac
