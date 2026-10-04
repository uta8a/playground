#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"

run_id="${1:?usage: check-alp-loki.sh RUN_ID}"
result_dir="results/$run_id"
source_log="$result_dir/access.log"
loki_port="${LOKI_PORT:-3101}"
range_seconds="${LOKI_RANGE_SECONDS:-600}"
temp_dir="$(mktemp -d)"
failures=0

cleanup() { rm -rf "$temp_dir"; }
trap cleanup EXIT

if [[ ! -s "$source_log" ]]; then
  echo "Missing alp input log: $source_log" >&2
  exit 2
fi

docker compose run --rm --no-deps -T \
  -v "$root_dir/$result_dir:/results:ro" \
  alp json --file=/results/access.log --format=csv > "$temp_dir/alp.csv"

query_scalar() {
  local query="$1" response
  response="$(curl --fail --silent --show-error --get \
    --data-urlencode "query=${query}" \
    "http://localhost:${loki_port}/loki/api/v1/query")"
  jq -er '.data.result | if length == 0 then "0" else .[0].value[1] end' <<<"$response"
}

check_value() {
  local label="$1" expected="$2" query="$3" tolerance="$4" actual
  actual="$(query_scalar "$query")"
  if awk -v expected="$expected" -v actual="$actual" -v tolerance="$tolerance" 'BEGIN { d=expected-actual; if (d < 0) d=-d; exit !(d <= tolerance) }'; then
    printf 'PASS %-14s expected=%s actual=%s\n' "$label" "$expected" "$actual"
  else
    printf 'FAIL %-14s expected=%s actual=%s\n' "$label" "$expected" "$actual" >&2
    failures=$((failures + 1))
  fi
}

echo "Comparing alp CSV with Grafana/Loki metric queries for run: $run_id"
echo "Loki range window: ${range_seconds}s"

# The CSV is the same alp aggregation as alp.txt, but is reliable to parse.
while IFS=, read -r count one_xx two_xx three_xx four_xx five_xx method uri min max sum avg p90 p95 p99 stddev min_body max_body sum_body avg_body; do
  selector="{service_name=\"nginx\"} | json | benchmark_run_id=\"${run_id}\" | method=\"${method}\" | uri=\"${uri}\""
  count_query="sum(count_over_time(${selector} [${range_seconds}s]))"

  printf '\n%s %s\n' "$method" "$uri"
  check_value COUNT "$count" "$count_query" 0
  check_value 1XX "$one_xx" "sum(count_over_time(${selector} | status=~\"1..\" [${range_seconds}s]))" 0
  check_value 2XX "$two_xx" "sum(count_over_time(${selector} | status=~\"2..\" [${range_seconds}s]))" 0
  check_value 3XX "$three_xx" "sum(count_over_time(${selector} | status=~\"3..\" [${range_seconds}s]))" 0
  check_value 4XX "$four_xx" "sum(count_over_time(${selector} | status=~\"4..\" [${range_seconds}s]))" 0
  check_value 5XX "$five_xx" "sum(count_over_time(${selector} | status=~\"5..\" [${range_seconds}s]))" 0
  check_value MIN "$min" "min(min_over_time(${selector} | unwrap response_time [${range_seconds}s]))" 0.0005
  check_value MAX "$max" "max(max_over_time(${selector} | unwrap response_time [${range_seconds}s]))" 0.0005
  check_value AVG "$avg" "sum(sum_over_time(${selector} | unwrap response_time [${range_seconds}s])) / ${count_query}" 0.0005
  check_value P90 "$p90" "max(quantile_over_time(0.90, ${selector} | unwrap response_time [${range_seconds}s]))" 0.0005
  check_value P95 "$p95" "max(quantile_over_time(0.95, ${selector} | unwrap response_time [${range_seconds}s]))" 0.0005
  check_value P99 "$p99" "max(quantile_over_time(0.99, ${selector} | unwrap response_time [${range_seconds}s]))" 0.0005
  check_value MIN_BODY "$min_body" "min(min_over_time(${selector} | unwrap body_bytes [${range_seconds}s]))" 0.0005
  check_value MAX_BODY "$max_body" "max(max_over_time(${selector} | unwrap body_bytes [${range_seconds}s]))" 0.0005
  check_value SUM_BODY "$sum_body" "sum(sum_over_time(${selector} | unwrap body_bytes [${range_seconds}s]))" 0.0005
  check_value AVG_BODY "$avg_body" "sum(sum_over_time(${selector} | unwrap body_bytes [${range_seconds}s])) / ${count_query}" 0.0005
done < <(tail -n +2 "$temp_dir/alp.csv")

if (( failures == 0 )); then
  echo "\nPASS: alp and the Loki queries used by Grafana agree."
else
  echo "\nFAIL: ${failures} values differ." >&2
  exit 1
fi
