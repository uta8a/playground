#!/bin/bash
set -euxo pipefail

gh api graphql -f query="mutation {
  createProjectV2Field(input: {
    projectId: \"$GITHUB_PROJECT_ID\"
    name: \"$GITHUB_PROJECT_CUSTOM_FIELD_NAME\"
    dataType: TEXT
  }) {
    projectV2Field {
      ... on ProjectV2Field {
        id
        name
        dataType 
      }
    }
  }
}"
