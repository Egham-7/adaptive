#!/bin/bash
# Monitor evaluation progress

echo "=========================================="
echo "Evaluation Progress Monitor"
echo "=========================================="
echo ""

echo "Prediction files created:"
ls -lth adaptive_router/data/unirouter/predictions/ 2>/dev/null || echo "No predictions directory"
echo ""

echo "Current LLM profiles:"
if [ -f adaptive_router/data/unirouter/clusters/llm_profiles.json ]; then
    python3 -c "
import json
with open('adaptive_router/data/unirouter/clusters/llm_profiles.json') as f:
    profiles = json.load(f)
    print(f'Total models profiled: {len(profiles)}')
    for model in sorted(profiles.keys()):
        errors = profiles[model]
        avg_error = sum(errors) / len(errors)
        print(f'  {model}: avg error {avg_error:.1%}')
" 2>/dev/null || echo "Could not read profiles"
else
    echo "No profiles file yet"
fi

echo ""
echo "=========================================="
