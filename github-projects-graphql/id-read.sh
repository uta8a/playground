#!/bin/bash
set -euxo pipefail

gh api graphql -f query="query {
  repository(owner: \"$GITHUB_OWNER\", name: \"$GITHUB_REPO\") {
    projectsV2(first: 10) {
      nodes {
        id
        title
      }
    }
  }
}"
