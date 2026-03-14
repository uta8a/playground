# Design: OpenRewrite codemod for replacing `Paths.get(..)` with `Path.of(..)`

## Goal
Create an OpenRewrite-based codemod library for a Java 26 / Gradle 9.4.0 project initialized with `gradle init`.

This codemod must:
- replace `java.nio.file.Paths.get(..)` with `java.nio.file.Path.of(..)`
- preserve arguments and semantics
- clean up imports automatically
- be testable and idempotent

## Scope
In scope:
- Gradle build setup for an OpenRewrite recipe project
- declarative recipe definition under `META-INF/rewrite/rewrite.yml`
- tests using RewriteTest
- sample code fixtures

Out of scope:
- custom Java visitor implementation unless strictly necessary
- composite Java migration recipes
- unrelated formatting/style changes
- non-Java source transformations

## Implementation strategy
Use the existing OpenRewrite recipe `org.openrewrite.java.migrate.nio.file.PathsGetToPathOf` and wrap it with an organization-specific recipe name.

Reason:
- this exact migration already exists upstream
- wrapping it minimizes implementation cost and risk
- we retain a stable internal recipe name for future composition/extensibility

## Project structure
- keep the project as a Gradle-based Java project
- add OpenRewrite recipe-library support
- create:
  - `src/main/resources/META-INF/rewrite/rewrite.yml`
  - `src/test/java/.../PathsGetToPathOfTest.java`

## Build configuration
Use:
- `java` plugin
- `org.openrewrite.build.recipe-library-base` plugin

Dependencies:
- OpenRewrite BOM
- `org.openrewrite.recipe:rewrite-migrate-java`
- `org.openrewrite:rewrite-test` for tests

Java:
- configure Gradle toolchain to Java 26

## Recipe design
Create a recipe:
- name: `com.yourorg.java.migrate.PathsGetToPathOf`
- display name: `Replace Paths.get with Path.of`

Its implementation should delegate to:
- `org.openrewrite.java.migrate.nio.file.PathsGetToPathOf`

## Test design
Write RewriteTest cases for:
1. simple single-argument replacement
2. multi-argument replacement
3. fully-qualified invocation replacement
4. import cleanup
5. non-target code remains unchanged

Also verify:
- running the recipe twice is idempotent
- no extra formatting-only changes are introduced

## Acceptance criteria
- `./gradlew test` passes
- `./gradlew rewriteRun` applies the transformation correctly
- a second `./gradlew rewriteRun` produces no new diff
- at least 4 before/after tests exist
- the recipe is activatable by its organization-specific name

## Notes
Prefer a thin wrapper recipe over a custom visitor.
Do not add unrelated recipes.
Keep the codemod single-purpose and review-friendly.
