#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Test runner for rfm_rl
#
# Usage:
#   ./tests/run_tests.sh              # run all tests
#   ./tests/run_tests.sh unit         # buffer & sampler unit tests only
#   ./tests/run_tests.sh integration  # CartPole integration tests only
#   ./tests/run_tests.sh fast         # unit tests, fail-fast
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."

case "${1:-all}" in
  unit)
    uv run python -m pytest tests/ -m "not integration" "$@"
    ;;
  integration)
    uv run python -m pytest tests/ -m integration "${@:2}"
    ;;
  fast)
    uv run python -m pytest tests/ -m "not integration" -x "${@:2}"
    ;;
  all)
    uv run python -m pytest tests/ "${@:2}"
    ;;
  *)
    echo "Usage: $0 {all|unit|integration|fast}"
    exit 1
    ;;
esac
