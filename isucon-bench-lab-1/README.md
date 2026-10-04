# ISUCON Bench Lab — Phase 1

`ab`、nginx の JSON access log、`alp` を使い、条件を固定した性能実験を繰り返すための最小環境です。OTel/Grafana は Phase 2 以降に追加します。

## What it does

`./scripts/bench.sh` は一つの benchmark run として、次を実行します。

1. run ID と現在の Git メタデータを `meta.json` に記録する
2. nginx を起動し、access log を空にする
3. 30 秒・concurrency 50・KeepAlive 有効で `ab` を実行する
4. client-side の `ab.txt` と percentile CSV を保存する
5. nginx の server-side access log を `alp` で URI ごとに集計する

結果は `results/<run-id>/` に保存されます。各 run は `X-Benchmark-Run-Id` ヘッダーで nginx の JSON log と対応付けられます。

## Prerequisites

- Docker Desktop（Compose v2 を含む）

`alp` は公式 GitHub Release の Linux バイナリ（v1.0.22）をビルド時にダウンロードし、公開されている checksum で検証します。実行イメージは `FROM scratch` で、`alp` バイナリだけを含みます。

## Run

```bash
cd isucon-bench-lab-1
./scripts/bench.sh
```

実験条件を明示的に変える場合:

```bash
CONCURRENCY=100 DURATION_SECONDS=60 PROFILE=static-c100-60s-keepalive ./scripts/bench.sh
```

生成されるファイル:

```text
results/<run-id>/
├── meta.json        # 条件と Git SHA
├── ab.txt           # client-side latency / RPS
├── percentile.csv   # 1〜100 percentile
└── alp.txt          # server-side request_time の URI 別集計
```

nginx は `http://localhost:8080/`、`stub_status` は `http://localhost:8080/status` で確認できます。`/status` はアクセスログから除外しています。

## Notes

`ab` と alp が示す遅延は同じではありません。`ab` はクライアントから見た待ち時間、alp は nginx の `$request_time`（JSON log 上では `response_time`）を集計します。この差分も性能調査の対象として残します。
