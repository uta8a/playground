#!/bin/bash
set -euxo pipefail

gh api graphql -f query="query {
  node(id: \"$GITHUB_PROJECT_ID\") {
    ... on ProjectV2 {
      fields(first: 20) {
        nodes {
          ... on ProjectV2Field {
            id
            name
            dataType  
          }
        }
      }
    }
  }
}"
