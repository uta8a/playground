#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"

concurrency="${CONCURRENCY:-50}"
duration_seconds="${DURATION_SECONDS:-30}"
keepalive="${KEEPALIVE:-true}"
nginx_port="${NGINX_PORT:-8081}"
profile="${PROFILE:-static-c${concurrency}-${duration_seconds}s-keepalive}"
run_id="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
result_dir="results/$run_id"

if ! [[ "$concurrency" =~ ^[1-9][0-9]*$ ]] || ! [[ "$duration_seconds" =~ ^[1-9][0-9]*$ ]]; then
  echo "CONCURRENCY and DURATION_SECONDS must be positive integers." >&2
  exit 2
fi

mkdir -p "$result_dir" logs
: > logs/access.log

git_sha="$(git rev-parse --short=12 HEAD 2>/dev/null || printf 'unknown')"
git_commit_time="$(git show -s --format=%cI HEAD 2>/dev/null || printf 'unknown')"

cat > "$result_dir/meta.json" <<EOF
{
  "run_id": "$run_id",
  "git_sha": "$git_sha",
  "git_commit_time": "$git_commit_time",
  "profile": "$profile",
  "concurrency": $concurrency,
  "duration_seconds": $duration_seconds,
  "keepalive": $keepalive
}
EOF

docker compose up -d nginx otel-agent lgtm

for _ in $(seq 1 30); do
  if curl --fail --silent "http://localhost:${nginx_port}/" >/dev/null; then
    break
  fi
  sleep 1
done
curl --fail --silent "http://localhost:${nginx_port}/" >/dev/null || {
  echo "nginx did not become ready." >&2
  exit 1
}

ab_args=(-n 10000000 -t "$duration_seconds" -c "$concurrency" -H "X-Benchmark-Run-Id: $run_id" -e /results/percentile.csv)
if [[ "$keepalive" == "true" ]]; then
  ab_args+=(-k)
fi
ab_args+=(http://nginx/)

docker compose run --rm --no-deps -v "$root_dir/$result_dir:/results" loadgen "${ab_args[@]}" | tee "$result_dir/ab.txt"

# Read the exact same records that OTel receives, but exclude readiness probes
# that do not have a benchmark_run_id.
jq -c --arg run_id "$run_id" 'select(.benchmark_run_id == $run_id)' logs/access.log > "$result_dir/access.log"
docker compose run --rm --no-deps -v "$root_dir/$result_dir:/results:ro" alp json --file=/results/access.log > "$result_dir/alp.txt"

./scripts/otel-verify.sh "$run_id" "$result_dir/access.log" > "$result_dir/otel-verify.txt"

echo "Benchmark complete: $result_dir"
