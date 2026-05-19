#!/usr/bin/env bash
set -euo pipefail

# Minimal launcher to bring up learner, rollout, and eval for Meta-World async training.
# - Pins each component to a GPU (configurable via env vars)
# - Waits indefinitely for the learner gRPC port before starting rollout/eval
# - Leaves learner in the foreground; rollout/eval run in the background

# Config (override via environment)
ENV_NAME=${ENV_NAME:-"Meta-World/MT1/drawer-close-v3"}
CHUNK_SIZE=${CHUNK_SIZE:-null}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-50051}
REWARD_RELABEL_HOST=${REWARD_RELABEL_HOST:-127.0.0.1}
REWARD_RELABEL_PORT=${REWARD_RELABEL_PORT:-50052}
NUM_OFFLINE_STEPS=${NUM_OFFLINE_STEPS:-0}
NUM_ROLLOUTS=${NUM_ROLLOUTS:-100000}

REWARD_RELABEL_GPU=${REWARD_RELABEL_GPU:-0}
LEARNER_GPU=${LEARNER_GPU:-0}
ROLLOUT_GPU=${ROLLOUT_GPU:-0}
EVAL_GPU=${EVAL_GPU:-0}

# Hydra config overrides (can be extended)
HYDRA_OVERRIDES=${HYDRA_OVERRIDES:-""}

# Resolve repository root
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

# Clean shutdown: propagate SIGINT/SIGTERM to children
PIDS=()
cleanup() {
  echo "[LAUNCH] Shutting down all processes..."
  for pid in "${PIDS[@]:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
  echo "[LAUNCH] All processes stopped."
}
trap cleanup INT TERM EXIT

# Wait for port helper function
wait_for_port() {
  local host=$1
  local port=$2
  local name=$3
  local max_attempts=60
  local attempt=0
  
  echo "[LAUNCH] Waiting for ${name} at ${host}:${port}..."
  while [ $attempt -lt $max_attempts ]; do
    # Try to connect to the port using timeout and bash's /dev/tcp
    if timeout 1 bash -c "echo >/dev/tcp/${host}/${port}" 2>/dev/null || \
       (command -v nc >/dev/null && nc -z "${host}" "${port}" 2>/dev/null); then
      echo "[LAUNCH] ${name} is ready at ${host}:${port}"
      return 0
    fi
    sleep 1
    attempt=$((attempt + 1))
  done
  echo "[LAUNCH] ERROR: ${name} did not become available at ${host}:${port} after ${max_attempts} seconds"
  return 1
}

echo "[LAUNCH] Starting reward relabel server on GPU ${REWARD_RELABEL_GPU} at ${REWARD_RELABEL_HOST}:${REWARD_RELABEL_PORT}"
CUDA_VISIBLE_DEVICES="${REWARD_RELABEL_GPU}" uv run \
  python scripts/start_reward_relabel_server.py \
    server.host="${REWARD_RELABEL_HOST}" \
    server.port="${REWARD_RELABEL_PORT}" \
    ${HYDRA_OVERRIDES} &
PIDS+=($!)

# Wait for reward relabel server to be ready
wait_for_port "${REWARD_RELABEL_HOST}" "${REWARD_RELABEL_PORT}" "Reward relabel server"

echo "[LAUNCH] Starting learner on GPU ${LEARNER_GPU} at ${HOST}:${PORT} (env=${ENV_NAME})"
CUDA_VISIBLE_DEVICES="${LEARNER_GPU}" uv run \
  python scripts/train_async.py \
    mode=learner \
    ~algorithm@offline_algorithm \
    algorithm@online_algorithm=sac \
    online_alg_name=sac \
    num_offline_steps="${NUM_OFFLINE_STEPS}" \
    num_rollouts="${NUM_ROLLOUTS}" \
    env_name="${ENV_NAME}" \
    chunk_size="${CHUNK_SIZE}" \
    distributed.learner_server.host="${HOST}" \
    distributed.learner_server.port="${PORT}" \
    distributed.learner_server.address="${HOST}:${PORT}" \
    distributed_reward_relabel.server_address="${REWARD_RELABEL_HOST}:${REWARD_RELABEL_PORT}" \
    ${HYDRA_OVERRIDES} &
PIDS+=($!)

# Wait for learner server to be ready
wait_for_port "${HOST}" "${PORT}" "Learner server"

echo "[LAUNCH] Starting rollout worker on GPU ${ROLLOUT_GPU}"
CUDA_VISIBLE_DEVICES="${ROLLOUT_GPU}" uv run \
  python scripts/train_async.py \
    mode=rollout \
    env_name="${ENV_NAME}" \
    chunk_size="${CHUNK_SIZE}" \
    distributed.learner_server.address="${HOST}:${PORT}" \
    ${HYDRA_OVERRIDES} &
PIDS+=($!)

echo "[LAUNCH] Starting eval worker on GPU ${EVAL_GPU}"
CUDA_VISIBLE_DEVICES="${EVAL_GPU}" uv run \
  python scripts/train_async.py \
    mode=eval \
    env_name="${ENV_NAME}" \
    chunk_size="${CHUNK_SIZE}" \
    distributed.learner_server.address="${HOST}:${PORT}" \
    ${HYDRA_OVERRIDES} &
PIDS+=($!)

echo "[LAUNCH] All components launched. PIDs: ${PIDS[*]}"
echo "[LAUNCH] Press Ctrl-C to stop all processes."

# Keep the script alive while children run
wait -n || true
wait || true