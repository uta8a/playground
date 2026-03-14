plugins {
    id("buildlogic.java-library-conventions")
    id("org.openrewrite.build.recipe-library-base") version "latest.release"
}

repositories {
    mavenCentral()
}

dependencies {
    implementation(platform("org.openrewrite.recipe:rewrite-recipe-bom:latest.release"))
    implementation("org.openrewrite.recipe:rewrite-migrate-java")

    testImplementation("org.openrewrite:rewrite-test")
    testImplementation("org.openrewrite:rewrite-java")
}
