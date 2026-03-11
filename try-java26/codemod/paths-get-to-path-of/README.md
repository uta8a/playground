# paths-get-to-path-of

`java.nio.file.Paths.get(..)` を `java.nio.file.Path.of(..)` に置き換えるための OpenRewrite レシピライブラリです。

- レシピ名: `com.yourorg.java.migrate.PathsGetToPathOf`
- 委譲先: `org.openrewrite.java.migrate.nio.file.PathsGetToPathOf`

## 前提条件

- Java 26+
- Gradle 9.x

## レシピライブラリのビルド

このモジュールを含むリポジトリのルートで実行します。

```bash
./gradlew :paths-get-to-path-of:build
```

生成される jar は次の場所です。

- `paths-get-to-path-of/build/libs/*.jar`

## 他の Gradle プロジェクトへ適用する

対象プロジェクト側で Rewrite プラグインを適用し、このレシピを有効化します。

### 方法1: ローカルのレシピ jar を直接参照する（最短）

1. このモジュールで生成した jar をコピー、またはパス参照できるようにします。
2. 対象プロジェクトの `build.gradle.kts` を次のように設定します。

```kotlin
plugins {
    id("org.openrewrite.rewrite") version "latest.release"
}

dependencies {
    // このモジュールで生成したレシピ jar を参照
    rewrite(files("/absolute/path/to/paths-get-to-path-of/build/libs/paths-get-to-path-of-<version>.jar"))
}

rewrite {
    activeRecipe("com.yourorg.java.migrate.PathsGetToPathOf")

    // 任意: Gradle Kotlin DSL や build 出力の解析時に起きる既知問題を回避
    exclusion("**/*.gradle.kts")
    exclusion("**/build/**")
}
```

3. 対象プロジェクトで codemod を実行します。

```bash
./gradlew rewriteRun --no-configuration-cache
```

4. 冪等性確認のため、もう一度実行します。

```bash
./gradlew rewriteRun --no-configuration-cache
```

### 方法2: Maven Local に公開して座標で参照する

1. 必要に応じてこのモジュールの `group` と `version` を設定します。
2. Maven Local に公開します。

```bash
./gradlew :paths-get-to-path-of:publishToMavenLocal
```

3. 対象プロジェクトの `build.gradle.kts` を次のように設定します。

```kotlin
plugins {
    id("org.openrewrite.rewrite") version "latest.release"
}

repositories {
    mavenLocal()
    mavenCentral()
}

dependencies {
    rewrite("com.yourorg:paths-get-to-path-of:<version>")
}

rewrite {
    activeRecipe("com.yourorg.java.migrate.PathsGetToPathOf")
    exclusion("**/*.gradle.kts")
    exclusion("**/build/**")
}
```

4. 実行します。

```bash
./gradlew rewriteRun --no-configuration-cache
```

## 期待される変換

変換前:

```java
import java.nio.file.Paths;

class Example {
    void run() {
        Paths.get("dir", "file.txt");
    }
}
```

変換後:

```java
import java.nio.file.Path;

class Example {
    void run() {
        Path.of("dir", "file.txt");
    }
}
```

## 対象プロジェクトでの確認チェックリスト

- `./gradlew rewriteRun --no-configuration-cache` が成功する。
- `Paths.get(..)` の呼び出しが `Path.of(..)` に変換される。
- import が整理される（`Paths` が削除され、必要に応じて `Path` が追加される）。
- 2回目の `rewriteRun` で追加差分が出ない。
