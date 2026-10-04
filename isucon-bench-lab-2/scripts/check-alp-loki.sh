#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"

run_id="${1:?usage: check-alp-loki.sh RUN_ID}"
result_dir="results/$run_id"
source_log="$result_dir/access.log"
loki_port="${LOKI_PORT:-3101}"
now_seconds="$(date -u +%s)"
start_ns="${LOKI_START_NS:-$(( (now_seconds - 300) * 1000000000 ))}"
end_ns="${LOKI_END_NS:-$(( (now_seconds + 30) * 1000000000 ))}"
query="{service_name=\"nginx\"} |= \"${run_id}\""
temp_dir="$(mktemp -d)"

cleanup() {
  rm -rf "$temp_dir"
}
trap cleanup EXIT

if [[ ! -s "$source_log" ]]; then
  echo "Missing alp input log: $source_log" >&2
  exit 2
fi

echo "Comparing run: $run_id"
echo "Loki query window: $start_ns .. $end_ns"

docker compose run --rm --no-deps -T \
  -v "$root_dir/$result_dir:/results:ro" \
  alp json --file=/results/access.log --format=csv > "$temp_dir/alp-source.csv"

curl --fail --silent --show-error --get \
  --data-urlencode "query=${query}" \
  --data-urlencode "start=${start_ns}" \
  --data-urlencode "end=${end_ns}" \
  "http://localhost:${loki_port}/loki/api/v1/query_range" > "$temp_dir/loki-response.json"

jq -e '.status == "success"' "$temp_dir/loki-response.json" >/dev/null
jq -r '.data.result[]?.values[]?[1]' "$temp_dir/loki-response.json" \
  | jq -c --arg run_id "$run_id" 'select(.benchmark_run_id == $run_id)' \
  > "$temp_dir/loki-access.log"

if [[ ! -s "$temp_dir/loki-access.log" ]]; then
  echo "No access logs found in Loki for this run." >&2
  echo "Wait for the OTel Collector, or widen the query window with LOKI_START_NS / LOKI_END_NS." >&2
  exit 1
fi

docker compose run --rm --no-deps -T \
  -v "$temp_dir:/comparison:ro" \
  alp json --file=/comparison/loki-access.log --format=csv > "$temp_dir/alp-loki.csv"

source_records="$(wc -l < "$source_log" | tr -d ' ')"
loki_records="$(wc -l < "$temp_dir/loki-access.log" | tr -d ' ')"
echo "alp source records: $source_records"
echo "Loki records:       $loki_records"

if diff -u "$temp_dir/alp-source.csv" "$temp_dir/alp-loki.csv"; then
  echo "PASS: alp and Loki produce identical CSV aggregations."
else
  echo "FAIL: alp aggregation differs from the logs Grafana reads from Loki." >&2
  exit 1
fi
