#!/bin/bash
set -euxo pipefail

gh api graphql -f query="query {
  node(id: \"$GITHUB_PROJECT_ID\") {
    ... on ProjectV2 {
      field(name: \"$GITHUB_PROJECT_CUSTOM_FIELD_NAME\") {
        ... on ProjectV2Field {
          id
          name
          dataType
        }
      }
    }
  }
}"
