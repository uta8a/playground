# Plan: Serverless TODO App (AWS CDK)
> ステータス: **完了済み** (2026-03-14)
## Goal
AWS CDKでデプロイできるシンプルなサーバレスTODOアプリを作る。  
CRUD API + DynamoDB + 静的フロントエンドを最小構成で実装する。
---
## Architecture
Frontend
- S3
- CloudFront
Backend
- API Gateway (HTTP API)
- Lambda
Database
- DynamoDB
IaC
- AWS CDK (TypeScript)
---
## Features
TODO fields
- id
- title
- completed
- createdAt
- updatedAt
API
- GET /todos
- POST /todos
- PATCH /todos/{id}
- DELETE /todos/{id}
Frontend
- TODO一覧表示
- TODO追加
- 完了切替
- 削除
---
## CDK Structure
Stack
- TodoAppStack
Constructs
- DatabaseConstruct
- ApiConstruct
- FrontendConstruct
Directory
```
bin/app.ts
lib/todo-app-stack.ts
lib/constructs/*
lambda/todo-api/*
frontend/*
```
---
## Resources
DynamoDB
- Table: TodoTable
- PK: id
- Billing: PAY_PER_REQUEST
Lambda
- TodoApiHandler
- CRUD処理
API Gateway
- HTTP API
- Lambda統合
- CORS有効
Frontend
- S3 bucket
- CloudFront distribution
---
## Destroy Strategy
`cdk destroy`で全リソース削除できることを必須とする。
全永続リソース
```
removalPolicy: DESTROY
```
S3
```
autoDeleteObjects: true
removalPolicy: DESTROY
```
CloudWatch Logs
```
removalPolicy: DESTROY
```
理由
- S3はオブジェクトがあると削除不可
- destroy失敗を防ぐ
---
## Implementation Steps
1. CDK TypeScript初期化
2. TodoAppStack作成
3. DynamoDB作成
4. Lambda CRUD実装
5. HTTP API接続
6. フロントエンド作成
7. S3 + CloudFront配置
8. cdk deploy
---
## Verification
```
cdk deploy
cdk destroy
```
destroy後に以下が存在しないこと
- DynamoDB
- S3
- CloudFront
- Lambda
- Logs
---
## Definition of Done
- CDK deploy成功
- TODO CRUD動作
- destroyで完全削除
- READMEにdeploy/destroy手順
