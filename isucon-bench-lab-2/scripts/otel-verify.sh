#!/usr/bin/env bash
set -euo pipefail

run_id="${1:?usage: otel-verify.sh RUN_ID ACCESS_LOG_PATH}"
access_log="${2:?usage: otel-verify.sh RUN_ID ACCESS_LOG_PATH}"
expected_count="$(wc -l < "$access_log" | tr -d ' ')"
query="{service_name=\"nginx\"} |= \"${run_id}\""
loki_port="${LOKI_PORT:-3101}"
grafana_port="${GRAFANA_PORT:-3001}"

echo "run_id: ${run_id}"
echo "alp input records: ${expected_count}"
echo "LogQL: ${query}"
echo
echo "Grafana Explore: http://localhost:${grafana_port}/explore"
echo "Wait a few seconds for the collector, then run:"
printf 'curl -G --data-urlencode %q http://localhost:%s/loki/api/v1/query_range\n' "query=${query}" "$loki_port"
