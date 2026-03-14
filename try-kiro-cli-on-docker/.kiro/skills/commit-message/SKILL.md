---
name: commit-message
description: Format high-quality Git commit messages from staged or unstaged changes with stable conventions. Use when Codex is asked to create, polish, or validate commit messages, especially for Conventional Commits, commitlint-compatible repositories, squash/fixup flows, or teams that require consistent subject/body/footer structure.
---

# Git Commit Formatter

## Overview
Generate a commit message that matches repository conventions and accurately summarizes code changes. Detect style first, then produce one final message candidate plus optional alternatives only when requested.

## Workflow
1. Detect the expected format from repository signals.
2. Summarize the real intent of the change from diffs, not filenames alone.
3. Produce a message with strict structure and length constraints.
4. Validate against local conventions and rewrite once if needed.

## Detect Format
Check in this order and follow the first clear signal:
1. `.git/COMMIT_EDITMSG` recent style (if available)
2. `commitlint.config.*`, `.commitlintrc*`, `lefthook.yml`, `.husky/*`
3. `package.json` scripts/tools mentioning commitlint or cz
4. Recent `git log --oneline` pattern

If no clear convention is detected, default to Conventional Commits.

## Build Message
Use this template unless repo rules differ:

```text
type(scope): subject

body

footer
```

Apply these rules:
- `type`: choose the smallest accurate type (`feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `build`, `ci`, `perf`, `style`)
- `scope`: include only if clear and stable (package/module/domain)
- `subject`: imperative, present tense, no trailing period, target <= 50 chars
- `body`: explain why and what changed; wrap around 72 chars
- `footer`: include `BREAKING CHANGE:` and issue links when applicable

## Mapping Heuristics
Use these defaults when intent is ambiguous:
- Bug correction with behavior change: `fix`
- New user-visible behavior: `feat`
- Behavior-preserving cleanup: `refactor`
- Test-only updates: `test`
- Tooling/dependency/build-only: `chore` or `build` or `ci`
- Comment/readme/docs-only: `docs`

Prefer correctness over verbosity. If multiple logical changes are mixed, propose split commits.

## Output Contract
Return:
1. Final commit message in a single fenced block.
2. Short rationale (1-3 lines) mapping diff to `type/scope`.
3. If uncertainty is high, ask one focused question; otherwise choose best default.

Do not include git commands unless explicitly requested.
