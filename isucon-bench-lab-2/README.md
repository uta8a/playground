# ISUCON Bench Lab — Phase 2

`ab`、nginx の JSON access log、`alp`、OpenTelemetry、Grafana/Loki を使い、条件を固定した性能実験を繰り返すための環境です。

## What it does

`./scripts/bench.sh` は一つの benchmark run として、次を実行します。

1. run ID と現在の Git メタデータを `meta.json` に記録する
2. nginx を起動し、access log を空にする
3. 30 秒・concurrency 50・KeepAlive 有効で `ab` を実行する
4. client-side の `ab.txt` と percentile CSV を保存する
5. `benchmark_run_id` が一致する access log だけを `alp` で URI ごとに集計する
6. 同じ access log を OTel Collector が Loki に転送し、run ごとの LogQL を保存する

結果は `results/<run-id>/` に保存されます。各 run は `X-Benchmark-Run-Id` ヘッダーで nginx の JSON log と対応付けられます。

## Prerequisites

- Docker Desktop（Compose v2 を含む）

`alp` は公式 GitHub Release の Linux バイナリ（v1.0.22）をビルド時にダウンロードし、公開されている checksum で検証します。実行イメージは `FROM scratch` で、`alp` バイナリだけを含みます。

## Run

```bash
cd isucon-bench-lab-2
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
├── access.log       # alp と OTel の対応確認に使う、run ID で絞った入力ログ
├── alp.txt          # server-side request_time の URI 別集計
└── otel-verify.txt  # Loki での照合用 LogQL と期待レコード数
```

既定では nginx は `http://localhost:8081/`、Grafana は `http://localhost:3001/`、Loki API は `http://localhost:3101/`、`stub_status` は `http://localhost:8081/status` です。`NGINX_PORT`、`GRAFANA_PORT`、`LOKI_PORT` で変更できます。`/status` はアクセスログから除外しています。

## alp と OTel の対応確認

各 run の `access.log` は `benchmark_run_id` で絞った nginx JSON log です。`alp.txt` はこのファイルのみを集計します。一方 OTel Collector は元の `logs/access.log` を JSON として読み、同じ `benchmark_run_id` を log attribute に付けて Loki へ送ります。

`otel-verify.txt` の LogQL を Grafana Explore で実行し、返るログ件数が `alp input records` と一致することを確認してください。その後、同じ run ID の `alp.txt` と Loki の各ログの `response_time`、`status`、`uri` を比較できます。Loki には readiness probe も入りますが、そのリクエストには run ID がないため、この照合からは除外されます。

## Grafana dashboard

[`grafana/alp-equivalent-dashboard.json`](grafana/alp-equivalent-dashboard.json) を Grafana の **Dashboards → New → Import** から import し、Loki datasource を選択してください。`Benchmark run ID` に `otel-verify.txt` の `run_id` を入力すると、alp table 相当の URI 別集計と、その根拠となる access log を表示します。

Grafana の time range は benchmark 実施時刻を含めてください。alp と同じ server-side 値を表示するため、`response_time` は秒、`body_bytes` は bytes です。`ab.txt` / `percentile.csv` の client-side latency とは直接一致しません。

### alp と Loki の集計一致チェック

Collector が Loki へ転送し終えた後、run ID を指定して実行します。

```bash
./scripts/check-alp-loki.sh 20261004T015834Z
```

スクリプトは `access.log` を alp CSV で集計し、Grafana dashboard と同じ Loki/LogQL のメトリクス集計 API へ問い合わせます。COUNT、各 HTTP ステータス帯、MIN/MAX/AVG/p90/p95/p99 latency、body bytes を比較します。数値は alp の小数第3位表示に合わせ、latency は 0.0005 秒の許容差を持たせます。

生ログの `query_range` 取得には件数上限があるため使用しません。既定では直近 10 分の LogQL range vector を使います。スクリプトは Collector が全ログを Loki へ転送するまで最大 60 秒待機します。転送に時間が掛かる場合は `LOKI_WAIT_SECONDS=180`、時間窓が足りない場合は `LOKI_RANGE_SECONDS=1800` を指定して再実行してください。

## Notes

`ab` と alp が示す遅延は同じではありません。`ab` はクライアントから見た待ち時間、alp は nginx の `$request_time`（JSON log 上では `response_time`）を集計します。この差分も性能調査の対象として残します。
