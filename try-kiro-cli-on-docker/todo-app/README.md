# TODO App (AWS CDK)

シンプルなサーバレスTODOアプリ。

## Architecture

- Frontend: S3 + CloudFront
- Backend: API Gateway (HTTP API) + Lambda
- Database: DynamoDB
- IaC: AWS CDK (TypeScript)

## Prerequisites

- Node.js 22+
- AWS CLI (configured)
- AWS CDK: `npm install -g aws-cdk`

## Deploy

```bash
# 初回のみ
cdk bootstrap

# デプロイ
cdk deploy
```

デプロイ完了後、出力される `FrontendUrl` にアクセスするとアプリが使えます。

## Destroy

```bash
cdk destroy
```

以下のリソースがすべて削除されます:
- DynamoDB Table
- S3 Bucket (オブジェクト含む)
- CloudFront Distribution
- Lambda Function
- CloudWatch Logs

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | /todos | 一覧取得 |
| POST | /todos | 作成 |
| PATCH | /todos/{id} | 更新 |
| DELETE | /todos/{id} | 削除 |
