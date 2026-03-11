package com.yourorg.java.migrate;

import org.junit.jupiter.api.Test;
import org.openrewrite.config.Environment;
import org.openrewrite.test.RecipeSpec;
import org.openrewrite.test.RewriteTest;

import static org.openrewrite.java.Assertions.java;

class PathsGetToPathOfTest implements RewriteTest {

    @Override
    public void defaults(RecipeSpec spec) {
        spec.recipe(Environment.builder()
                .scanRuntimeClasspath()
                .build()
                .activateRecipes("com.yourorg.java.migrate.PathsGetToPathOf"));
    }

    @Test
    void replacesSingleArgumentInvocation() {
        rewriteRun(
                java(
                        """
                                import java.nio.file.Paths;

                                class Test {
                                    void run() {
                                        Paths.get("a.txt");
                                    }
                                }
                                """,
                        """
                                import java.nio.file.Path;

                                class Test {
                                    void run() {
                                        Path.of("a.txt");
                                    }
                                }
                                """));
    }

    @Test
    void replacesMultiArgumentInvocation() {
        rewriteRun(
                java(
                        """
                                import java.nio.file.Paths;

                                class Test {
                                    void run() {
                                        Paths.get("dir", "file.txt");
                                    }
                                }
                                """,
                        """
                                import java.nio.file.Path;

                                class Test {
                                    void run() {
                                        Path.of("dir", "file.txt");
                                    }
                                }
                                """));
    }

    @Test
    void replacesFullyQualifiedInvocation() {
        rewriteRun(
                java(
                        """
                                class Test {
                                    void run() {
                                        java.nio.file.Paths.get("x", "y");
                                    }
                                }
                                """,
                        """
                                import java.nio.file.Path;

                                class Test {
                                    void run() {
                                        Path.of("x", "y");
                                    }
                                }
                                """));
    }

    @Test
    void cleansUpImports() {
        rewriteRun(
                java(
                        """
                                import java.nio.file.Paths;

                                class Test {
                                    void run() {
                                        Paths.get("x");
                                    }
                                }
                                """,
                        """
                                import java.nio.file.Path;

                                class Test {
                                    void run() {
                                        Path.of("x");
                                    }
                                }
                                """));
    }

    @Test
    void leavesNonTargetCodeUnchanged() {
        rewriteRun(
                java(
                        """
                                import java.nio.file.Path;

                                class Test {
                                    void run() {
                                        Path.of("already", "ok");
                                    }
                                }
                                """));
    }

    @Test
    void isIdempotentOnSecondCycle() {
        rewriteRun(
                spec -> spec.cycles(2).expectedCyclesThatMakeChanges(1),
                java(
                        """
                                import java.nio.file.Paths;

                                class Test {
                                    void run() {
                                        Paths.get("again");
                                    }
                                }
                                """,
                        """
                                import java.nio.file.Path;

                                class Test {
                                    void run() {
                                        Path.of("again");
                                    }
                                }
                                """));
    }
}
