#!/bin/bash
# Benchmark script for Go LLM API using GenAI-Perf

set -e

# Configuration
export ROUTER_URL="${ROUTER_URL:-localhost:8080}"
export MODEL_NAME="${MODEL_NAME:-adaptive-go-api}"
export WORKDIR="/workdir"
export RESULTS_DIR="${WORKDIR}/results"

# Ensure results directory exists
mkdir -p "$RESULTS_DIR"

# Test scenarios tailored for LLM API
declare -A testCases
testCases["Quick_Chat"]="50/25"
testCases["Code_Generation"]="100/200"
testCases["Text_Summarization"]="300/150"
testCases["Question_Answering"]="75/100"
testCases["Long_Form_Content"]="200/400"

# Function to check if the Go API is running
checkAPIHealth() {
  echo "Checking if Go API is accessible at $ROUTER_URL..."
  if curl -s -f "http://$ROUTER_URL/health" >/dev/null 2>&1; then
    echo "✓ Go API is accessible"
    return 0
  else
    echo "✗ Go API is not accessible at $ROUTER_URL"
    echo "Please ensure your Go API is running and accessible"
    return 1
  fi
}

# Function to run benchmarks with different configurations
runBenchmark() {
  local description="$1"
  local lengths="${testCases[$description]}"
  IFS='/' read -r inputLength outputLength <<<"$lengths"

  echo "=== Benchmarking: $description ==="
  echo "Input: $inputLength tokens, Output: $outputLength tokens"

  # Create test-specific results directory
  local testDir="$RESULTS_DIR/${description}"
  mkdir -p "$testDir"

  # Test different concurrency levels
  for concurrency in 1 2 5 10 20 50; do
    echo "Testing concurrency: $concurrency"

    local outputFile="${testDir}/${description}_c${concurrency}"

    # Run GenAI-Perf with OpenAI-compatible endpoint
    genai-perf profile \
      -m "$MODEL_NAME" \
      --endpoint-type chat \
      --service-kind openai \
      --streaming \
      -u "$ROUTER_URL" \
      --synthetic-input-tokens-mean "$inputLength" \
      --synthetic-input-tokens-stddev 10 \
      --concurrency "$concurrency" \
      --output-tokens-mean "$outputLength" \
      --output-tokens-stddev 20 \
      --num-prompts 50 \
      --warmup-mode count \
      --num-prompts-warmup 5 \
      --extra-inputs max_tokens:$((outputLength + 50)) \
      --extra-inputs temperature:0.7 \
      --extra-inputs top_p:0.9 \
      --measurement-interval 10000 \
      --profile-export-file "${outputFile}.json" \
      --artifact-dir "$testDir/artifacts_c${concurrency}" \
      -- \
      -v \
      --max-threads=128 \
      --collect-metrics \
      --metrics-url=http://localhost:8002/metrics || {
      echo "Warning: GenAI-Perf failed for concurrency $concurrency in $description"
      echo "Continuing with next test..."
      continue
    }

    echo "✓ Completed concurrency $concurrency for $description"
    echo "---"

    # Small delay between tests to avoid overwhelming the API
    sleep 2
  done

  echo "✓ Completed all concurrency tests for $description"
  echo "================================================="
}

# Function to run load test scenarios
runLoadTest() {
  echo "=== Running Load Test Scenarios ==="

  # Sustained load test
  echo "Running sustained load test..."
  genai-perf profile \
    -m "$MODEL_NAME" \
    --endpoint-type chat \
    --service-kind openai \
    -u "$ROUTER_URL" \
    --synthetic-input-tokens-mean 100 \
    --concurrency 10 \
    --output-tokens-mean 100 \
    --num-prompts 200 \
    --measurement-interval 30000 \
    --profile-export-file "$RESULTS_DIR/sustained_load.json" \
    --artifact-dir "$RESULTS_DIR/sustained_load_artifacts" \
    -- \
    -v \
    --max-threads=64

  echo "✓ Sustained load test completed"
}

# Function to run simple benchmark without tokenizer dependencies
runSimpleBenchmark() {
  echo "=== Running Simple Benchmark (No Tokenizer) ==="

  local test_scenarios=("quick_response:50" "medium_response:150" "long_response:300")

  for scenario in "${test_scenarios[@]}"; do
    IFS=':' read -r test_name max_tokens <<<"$scenario"

    echo "Testing: $test_name (max_tokens: $max_tokens)"

    for concurrency in 1 5 10 25; do
      echo "  Concurrency: $concurrency"

      genai-perf profile \
        -m "$MODEL_NAME" \
        --endpoint-type chat \
        --service-kind openai \
        -u "$ROUTER_URL" \
        --num-prompts 30 \
        --concurrency "$concurrency" \
        --extra-inputs max_tokens:"$max_tokens" \
        --extra-inputs temperature:0.7 \
        --measurement-interval 8000 \
        --profile-export-file "$RESULTS_DIR/simple_${test_name}_c${concurrency}.json" \
        --artifact-dir "$RESULTS_DIR/simple_${test_name}_c${concurrency}_artifacts" \
        -- \
        -v \
        --max-threads=64 || {
        echo "Warning: Simple benchmark failed for $test_name at concurrency $concurrency"
        continue
      }
    done

    echo "✓ Completed $test_name"
  done
}

# Main execution
main() {
  echo "Starting Go LLM API Benchmarking with GenAI-Perf"
  echo "================================================="
  echo "Router URL: $ROUTER_URL"
  echo "Model Name: $MODEL_NAME"
  echo "Results Directory: $RESULTS_DIR"
  echo "================================================="

  # Check API health
  if ! checkAPIHealth; then
    echo "Exiting due to API health check failure"
    exit 1
  fi

  # Run benchmarks based on available mode
  echo "Choose benchmark mode:"
  echo "1. Full benchmark with synthetic tokens (requires tokenizer)"
  echo "2. Simple benchmark (no tokenizer required)"
  echo "3. Load test only"
  echo "4. All tests"

  # Default to simple benchmark if no input
  local mode="${1:-2}"

  case $mode in
  1)
    echo "Running full benchmark..."
    for description in "${!testCases[@]}"; do
      runBenchmark "$description"
    done
    ;;
  2)
    echo "Running simple benchmark..."
    runSimpleBenchmark
    ;;
  3)
    echo "Running load test only..."
    runLoadTest
    ;;
  4)
    echo "Running all tests..."
    for description in "${!testCases[@]}"; do
      runBenchmark "$description"
    done
    runSimpleBenchmark
    runLoadTest
    ;;
  *)
    echo "Invalid mode. Running simple benchmark..."
    runSimpleBenchmark
    ;;
  esac

  echo ""
  echo "==============================================="
  echo "Benchmarking Complete!"
  echo "Results saved in: $RESULTS_DIR"
  echo "==============================================="
  echo ""
  echo "To analyze results, run:"
  echo "  python scripts/analyze_results.py"
  echo ""
}

# Run main function with all arguments
main "$@"

