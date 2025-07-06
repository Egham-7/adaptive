#!/bin/bash
# Complete benchmarking workflow for Go LLM API

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ROUTER_URL="${ROUTER_URL:-localhost:8080}"
MODEL_NAME="${MODEL_NAME:-adaptive-go-api}"
BENCHMARK_MODE="${BENCHMARK_MODE:-2}"  # Default to simple benchmark

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# Function to check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    print_status "✓ Docker is running"
    
    # Check if docker-compose is available
    if ! command -v docker-compose >/dev/null 2>&1; then
        print_error "docker-compose is not installed. Please install docker-compose."
        exit 1
    fi
    print_status "✓ docker-compose is available"
    
    # Check if Go API is accessible
    print_status "Checking Go API accessibility at $ROUTER_URL..."
    if curl -s -f "http://$ROUTER_URL/health" >/dev/null 2>&1; then
        print_status "✓ Go API is accessible"
    else
        print_warning "Go API is not accessible at $ROUTER_URL"
        print_warning "Make sure your Go API is running before starting benchmarks"
        
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_status "Exiting..."
            exit 1
        fi
    fi
}

# Function to setup environment
setup_environment() {
    print_header "Setting Up Environment"
    
    # Create necessary directories
    mkdir -p results/plots
    print_status "✓ Created results directories"
    
    # Export environment variables
    export ROUTER_URL
    export MODEL_NAME
    print_status "✓ Environment variables set"
    print_status "  ROUTER_URL: $ROUTER_URL"
    print_status "  MODEL_NAME: $MODEL_NAME"
}

# Function to start GenAI-Perf container
start_container() {
    print_header "Starting GenAI-Perf Container"
    
    # Stop any existing container
    docker-compose down >/dev/null 2>&1 || true
    
    # Start the container
    print_status "Starting GenAI-Perf container..."
    docker-compose up -d
    
    # Wait for container to be ready
    print_status "Waiting for container to be ready..."
    sleep 5
    
    # Verify container is running
    if docker-compose ps | grep -q "Up"; then
        print_status "✓ GenAI-Perf container is running"
    else
        print_error "Failed to start GenAI-Perf container"
        exit 1
    fi
}

# Function to run benchmarks
run_benchmarks() {
    print_header "Running Benchmarks"
    
    print_status "Starting benchmark execution..."
    print_status "Mode: $BENCHMARK_MODE"
    
    # Install required Python packages in container
    print_status "Installing Python dependencies..."
    docker-compose exec -T genai-perf pip install matplotlib seaborn pandas numpy >/dev/null 2>&1
    
    # Run the benchmark script
    print_status "Executing benchmark script..."
    if docker-compose exec -T genai-perf bash scripts/benchmark_go_api.sh "$BENCHMARK_MODE"; then
        print_status "✓ Benchmarks completed successfully"
    else
        print_error "Benchmark execution failed"
        return 1
    fi
}

# Function to analyze results
analyze_results() {
    print_header "Analyzing Results"
    
    print_status "Running results analysis..."
    if docker-compose exec -T genai-perf python scripts/analyze_results.py; then
        print_status "✓ Results analysis completed"
    else
        print_warning "Results analysis failed, but benchmark data is still available"
    fi
}

# Function to copy results to host
copy_results() {
    print_header "Copying Results to Host"
    
    # Copy all results from container to host
    print_status "Copying results from container..."
    docker-compose exec -T genai-perf cp -r /workdir/results /workdir/host_results >/dev/null 2>&1 || true
    
    # List available results
    print_status "Available results:"
    if [ -d "results" ]; then
        ls -la results/
        print_status "✓ Results copied to local ./results directory"
    else
        print_warning "No results directory found"
    fi
}

# Function to cleanup
cleanup() {
    print_header "Cleaning Up"
    
    print_status "Stopping GenAI-Perf container..."
    docker-compose down
    
    print_status "✓ Cleanup completed"
}

# Function to display results summary
display_summary() {
    print_header "Benchmark Summary"
    
    echo -e "${GREEN}Benchmarking Complete!${NC}"
    echo ""
    echo "Results are available in:"
    echo "  📊 Plots: ./results/plots/"
    echo "  📄 Summary: ./results/benchmark_summary_report.txt"
    echo "  📊 CSV Data: ./results/go_api_benchmark_results.csv"
    echo ""
    echo "To view results:"
    echo "  cat ./results/benchmark_summary_report.txt"
    echo "  open ./results/plots/"
    echo ""
    echo "To run analysis again:"
    echo "  docker-compose exec genai-perf python scripts/analyze_results.py"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -u, --url URL          Router URL (default: localhost:8080)"
    echo "  -m, --model MODEL      Model name (default: adaptive-go-api)"
    echo "  -b, --benchmark MODE   Benchmark mode (default: 2)"
    echo "                         1: Full benchmark with synthetic tokens"
    echo "                         2: Simple benchmark (no tokenizer)"
    echo "                         3: Load test only"
    echo "                         4: All tests"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run with defaults"
    echo "  $0 -u localhost:8080 -m my-model     # Custom URL and model"
    echo "  $0 -b 4                               # Run all benchmark modes"
    echo ""
}

# Main execution function
main() {
    print_header "Go LLM API Benchmarking with GenAI-Perf"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -u|--url)
                ROUTER_URL="$2"
                shift 2
                ;;
            -m|--model)
                MODEL_NAME="$2"
                shift 2
                ;;
            -b|--benchmark)
                BENCHMARK_MODE="$2"
                shift 2
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Execute workflow
    check_prerequisites
    setup_environment
    start_container
    
    # Run benchmarks with error handling
    if run_benchmarks; then
        analyze_results
        copy_results
        display_summary
    else
        print_error "Benchmarking failed"
        copy_results  # Still try to copy partial results
    fi
    
    cleanup
}

# Trap to ensure cleanup on exit
trap cleanup EXIT

# Run main function with all arguments
main "$@"