#!/bin/bash

# Script to create PR from dev to main
set -e

# Validate arguments
if [ $# -ne 4 ]; then
    echo "Error: This script requires exactly 4 arguments" >&2
    echo "Usage: $0 <commits_ahead> <merge_type> <actor> <workflow>" >&2
    exit 1
fi

COMMITS_AHEAD="$1"
MERGE_TYPE="$2"
ACTOR="$3"
WORKFLOW="$4"

# Check that all arguments are non-empty
if [ -z "$COMMITS_AHEAD" ]; then
    echo "Error: commits_ahead argument cannot be empty" >&2
    exit 1
fi

if [ -z "$MERGE_TYPE" ]; then
    echo "Error: merge_type argument cannot be empty" >&2
    exit 1
fi

if [ -z "$ACTOR" ]; then
    echo "Error: actor argument cannot be empty" >&2
    exit 1
fi

if [ -z "$WORKFLOW" ]; then
    echo "Error: workflow argument cannot be empty" >&2
    exit 1
fi

# Create secure temporary file
PR_BODY_FILE=$(mktemp)

# Set up cleanup trap
cleanup() {
    rm -f "$PR_BODY_FILE"
}
trap cleanup EXIT INT TERM

# Create PR body
PR_TITLE="🚀 Merge dev to main (${COMMITS_AHEAD} commits)"

# Create PR body as a temp file
cat > "$PR_BODY_FILE" << EOF
## 🚀 Automated Merge from Dev to Main

This PR was automatically created to merge the latest changes from \`dev\` to \`main\`.

### Changes Summary
- ${COMMITS_AHEAD} commits ahead of main

### Merge Details
- **Merge Type**: ${MERGE_TYPE}
- **Triggered by**: @${ACTOR}
- **Workflow**: ${WORKFLOW}

---
🤖 This PR will auto-merge once all checks pass.
EOF

# Check if PR already exists
EXISTING_PR=$(gh pr list --base main --head dev --json number --jq '.[0].number' 2>/dev/null || echo "")

if [ -n "$EXISTING_PR" ]; then
    echo "PR already exists: #$EXISTING_PR"
    echo "pr_number=$EXISTING_PR"
else
    # Create new PR
    NEW_PR=$(gh pr create \
        --title "$PR_TITLE" \
        --body-file "$PR_BODY_FILE" \
        --base main \
        --head dev \
        --assignee "$ACTOR")
    
    PR_NUMBER=$(echo "$NEW_PR" | grep -o '#[0-9]*' | sed 's/#//')
    echo "pr_number=$PR_NUMBER"
    echo "Created PR #$PR_NUMBER"
fi

# Cleanup handled by trap