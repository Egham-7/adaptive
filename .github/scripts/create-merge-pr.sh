#!/bin/bash

# Script to create PR from dev to main
set -e

COMMITS_AHEAD="$1"
MERGE_TYPE="$2"
ACTOR="$3"
WORKFLOW="$4"

# Create PR body
PR_TITLE="🚀 Merge dev to main (${COMMITS_AHEAD} commits)"

# Create PR body as a temp file
cat > /tmp/pr_body.md << EOF
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
        --body-file /tmp/pr_body.md \
        --base main \
        --head dev \
        --assignee "$ACTOR")
    
    PR_NUMBER=$(echo "$NEW_PR" | grep -o '#[0-9]*' | sed 's/#//')
    echo "pr_number=$PR_NUMBER"
    echo "Created PR #$PR_NUMBER"
fi

# Clean up
rm -f /tmp/pr_body.md