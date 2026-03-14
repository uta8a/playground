# Plan: Naming and Tagging Review
> ステータス: **完了済み** (2026-03-14)

## 目的
CDKコードのリソース命名・タグ付けの規約違反を修正し、運用・コスト管理・トレーサビリティを向上させる。

---

## 現状の問題点

### Naming

| 場所 | 現状 | 問題 |
|------|------|------|
| `bin/todo-app.ts` | `new TodoAppStack(app, "TodoAppStack")` | Stack IDとクラス名が同一で冗長 |
| `constructs/api.ts` | `logGroupName: "/aws/lambda/TodoApiHandler"` | ハードコードされた固定名。Lambda関数名と乖離リスクあり |
| `constructs/api.ts` | `new HttpLambdaIntegration("LambdaIntegration", fn)` | 汎用的すぎるID |
| `constructs/frontend.ts` | `new CfnOutput(scope, "FrontendUrl", ...)` | `scope`（Stack）に直接出力しており、Construct内で`this`を使うべき |
| Construct IDs全般 | `"Database"`, `"Api"`, `"Frontend"` | 短すぎて複数環境展開時に衝突リスク |

### Tagging

| 場所 | 問題 |
|------|------|
| `bin/todo-app.ts` | Stackにタグが一切付与されていない |
| 全リソース | `Project`, `Environment`, `ManagedBy` などの必須タグが存在しない |
| CloudWatch LogGroup | タグなし |

---

## 修正計画

### Step 1: Stack レベルのタグ付け追加

`bin/todo-app.ts` にて `cdk.Tags.of(app)` でアプリ全体にタグを付与する。

```typescript
// bin/todo-app.ts
const app = new cdk.App();
const stack = new TodoAppStack(app, "TodoApp");

cdk.Tags.of(stack).add("Project", "todo-app");
cdk.Tags.of(stack).add("ManagedBy", "cdk");
cdk.Tags.of(stack).add("Environment", app.node.tryGetContext("env") ?? "dev");
```

### Step 2: LogGroup名をLambda関数名から動的生成

`constructs/api.ts` にてハードコードを排除する。

```typescript
// 修正前
logGroupName: "/aws/lambda/TodoApiHandler",

// 修正後
logGroupName: `/aws/lambda/${fn.functionName}`,
```

ただし LogGroup は Lambda より先に作成する必要があるため、Lambda に `functionName` を明示指定する。

```typescript
const fn = new Function(this, "TodoApiHandler", {
  functionName: "todo-api-handler",
  // ...
});

new LogGroup(this, "LambdaLogs", {
  logGroupName: `/aws/lambda/${fn.functionName}`,
  // ...
});
```

### Step 3: CfnOutput のスコープ修正

`constructs/frontend.ts` の `CfnOutput` を `scope` から `this` に変更する。

```typescript
// 修正前
new CfnOutput(scope, "FrontendUrl", { ... });

// 修正後
new CfnOutput(this, "FrontendUrl", { ... });
```

### Step 4: HttpLambdaIntegration の ID を具体化

```typescript
// 修正前
new HttpLambdaIntegration("LambdaIntegration", fn)

// 修正後
new HttpLambdaIntegration("TodoApiLambdaIntegration", fn)
```

---

## 修正対象ファイル

- `bin/todo-app.ts` — タグ付け追加
- `lib/constructs/api.ts` — LogGroup名の動的化、Integration ID修正
- `lib/constructs/frontend.ts` — CfnOutput スコープ修正

---

## 優先度

| 修正 | 優先度 | 理由 |
|------|--------|------|
| Stackタグ付け | 高 | コスト配分・運用トレーサビリティに直結 |
| CfnOutput スコープ修正 | 高 | バグ（二重定義エラーの潜在リスク） |
| LogGroup名の動的化 | 中 | 関数名変更時の乖離防止 |
| Integration ID具体化 | 低 | 可読性改善 |

---

## Definition of Done

- `cdk synth` が警告なく通る
- 全リソースに `Project`, `Environment`, `ManagedBy` タグが伝播している（`cdk synth` のテンプレートで確認）
- LogGroup名がLambda関数名と一致している
- `CfnOutput` が正しいスコープで定義されている
