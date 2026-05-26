#!/bin/bash

# =============================================================================
# deploy_vllm.sh
# Concurrently deploys vllm on a list of remote hosts.
#
# Usage:
#   ./deploy_vllm.sh [OPTIONS]
#
# Options:
#   -m, --model       MODEL_NAME          Model to serve (required)
#   -t, --tp          TENSOR_PARALLEL     Tensor parallel size  (default: 4)
#   -d, --dp          DATA_PARALLEL       Data parallel size    (default: 2)
#   -p, --port        BASE_PORT           Base port             (default: 30600)
#   -g, --gpus        GPU_LIST            Comma-separated GPU indices (default: 0,1,2,3,4,5,6,7)
#   -H, --hosts       HOST1,HOST2,...     Comma-separated host list (overrides hardcoded list)
#   -u, --gpu-mem     GPU_MEM_UTIL        GPU memory utilisation (default: 0.90)
#   -e, --venv        VENV_PATH           Path to venv activate script (required)
#       --extra       "EXTRA_FLAGS"       Additional vllm flags appended after common flags
#   -h, --help                            Show this help and exit
#
# Common flags (always applied, not overridable):
#   vllm flags : --enable-prefix-caching --trust-remote-code --generation-config vllm --async-scheduling
#   env vars   : VLLM_SERVER_DEV_MODE=1
#
# Examples:
#   # Two data-parallel shards, one per host, each using all 8 GPUs:
#   ./deploy_vllm.sh \
#       -m openai/gpt-oss-120b \
#       -t 4 -d 2 \
#       -p 30600 \
#       -e /fsxnew/user/vllmenv/bin/activate
#
#   # Single shard on 4 GPUs, custom port and extra flags:
#   ./deploy_vllm.sh \
#       -m meta-llama/Llama-3-70b \
#       -t 4 -d 1 \
#       -p 8000 \
#       -g 0,1,2,3 \
#       -H ip-10-0-1-1,ip-10-0-1-2 \
#       -e /home/ubuntu/venv/bin/activate \
#       --extra "--max-num-seqs 64 --max-model-len 32768"
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/deployment_errors.log"

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL=""
TP=4
DP=2
BASE_PORT=30600
GPU_LIST="0,1,2,3,4,5,6,7"
GPU_MEM_UTIL=0.90
VENV_PATH=""
EXTRA_FLAGS=""

# ── Common settings (always applied) ─────────────────────────────────────────
COMMON_ENV="VLLM_SERVER_DEV_MODE=1"
COMMON_FLAGS="--enable-prefix-caching --trust-remote-code --generation-config vllm --async-scheduling"

HOSTS=(
  ip-10-0-249-61
  ip-10-0-252-170
)

# ── Argument parsing ──────────────────────────────────────────────────────────
usage() {
  sed -n '/^# Usage:/,/^# =====/p' "$0" | sed 's/^# \?//'
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model)      MODEL="$2";                       shift 2 ;;
    -t|--tp)         TP="$2";                           shift 2 ;;
    -d|--dp)         DP="$2";                           shift 2 ;;
    -p|--port)       BASE_PORT="$2";                    shift 2 ;;
    -g|--gpus)       GPU_LIST="$2";                     shift 2 ;;
    -H|--hosts)      IFS=',' read -ra HOSTS <<< "$2";  shift 2 ;;
    -u|--gpu-mem)    GPU_MEM_UTIL="$2";                 shift 2 ;;
    -e|--venv)       VENV_PATH="$2";                    shift 2 ;;
       --extra)      EXTRA_FLAGS="$2";                  shift 2 ;;  # appended after COMMON_FLAGS
    -h|--help)       usage ;;
    *) echo "Unknown option: $1" >&2; usage ;;
  esac
done

# ── Validation ────────────────────────────────────────────────────────────────
errors=()
[[ -z "$MODEL"     ]] && errors+=("--model is required.")
[[ -z "$VENV_PATH" ]] && errors+=("--venv (path to activate script) is required.")
[[ ! "$TP"   =~ ^[0-9]+$ ]] && errors+=("--tp must be a positive integer.")
[[ ! "$DP"   =~ ^[0-9]+$ ]] && errors+=("--dp must be a positive integer.")
[[ ! "$BASE_PORT" =~ ^[0-9]+$ ]] && errors+=("--port must be a positive integer.")

if [[ ${#errors[@]} -gt 0 ]]; then
  echo "ERROR: Invalid arguments:" >&2
  for e in "${errors[@]}"; do echo "  • $e" >&2; done
  echo "" >&2
  usage
fi

# ── Derived values ────────────────────────────────────────────────────────────
# Build CUDA_VISIBLE_DEVICES and one port per host by offsetting BASE_PORT.
# Each host gets a single vllm process using all specified GPUs.
SSH="ssh -o StrictHostKeyChecking=accept-new \
         -o BatchMode=yes \
         -o ConnectTimeout=15 \
         -o ServerAliveInterval=30 \
         -o ServerAliveCountMax=3"

echo "================================================================="
echo " vllm concurrent deployment — $(date)"
printf " Model  : %s\n"  "$MODEL"
printf " TP     : %s  DP: %s\n" "$TP" "$DP"
printf " GPUs   : %s\n"  "$GPU_LIST"
printf " Port   : %s (base, +1 per host)\n" "$BASE_PORT"
printf " Hosts  : %s\n"  "${#HOSTS[@]}"
printf " Env    : %s\n"  "$COMMON_ENV"
printf " Flags  : %s\n"  "$COMMON_FLAGS"
[[ -n "$EXTRA_FLAGS" ]] && printf " Extra  : %s\n" "$EXTRA_FLAGS"
printf " Log    : %s\n"  "$LOG_FILE"
echo "================================================================="

# ── Per-host deployment (concurrent) ─────────────────────────────────────────
host_index=0
for HOST in "${HOSTS[@]}"; do
  (
    PORT=$(( BASE_PORT + host_index ))
    SESSION="vllm_${HOST}"

    CMD="source ${VENV_PATH}; \
${COMMON_ENV} \
CUDA_VISIBLE_DEVICES=${GPU_LIST} \
uv run vllm serve ${MODEL} \
  --tensor-parallel-size ${TP} \
  --data-parallel-size ${DP} \
  --port ${PORT} \
  --gpu-memory-utilization ${GPU_MEM_UTIL} \
  ${COMMON_FLAGS} \
  ${EXTRA_FLAGS}"

    LOG_REMOTE="~/vllm_$(echo "$MODEL" | tr '/' '_')_$(date +%Y%m%d_%H%M%S).log"

    echo "[$(date '+%H:%M:%S')] Deploying → ${HOST}  (port ${PORT})"

    # ── SSH connectivity check ────────────────────────────────────────────────
    CHECK=$(eval "$SSH" "$HOST" "echo ok" 2>/dev/null || true)
    if [[ "$CHECK" != "ok" ]]; then
      ERR=$(eval "$SSH" "$HOST" "echo ok" 2>&1 || true)
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEPLOYMENT ERROR | host=${HOST} | ssh-check failed: ${ERR}" \
        | tee -a "$LOG_FILE"
      exit 1
    fi

    # ── Kill stale session ────────────────────────────────────────────────────
    eval "$SSH" "$HOST" "tmux kill-session -t '${SESSION}' 2>/dev/null; true" 2>/dev/null || true

    # ── Launch tmux session ───────────────────────────────────────────────────
    QUOTED_CMD=$(printf '%q' "${CMD}")
    ERR=$(eval "$SSH" "$HOST" \
      "tmux new-session -d -s '${SESSION}' -x 220 -y 50 \
        'bash -lc ${QUOTED_CMD} 2>&1 | tee ${LOG_REMOTE}'" 2>&1) || {
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEPLOYMENT ERROR | host=${HOST} port=${PORT} | tmux new-session failed: ${ERR}" \
        | tee -a "$LOG_FILE"
      exit 1
    }

    echo "[$(date '+%H:%M:%S')] ✓ ${HOST} — session '${SESSION}' started on port ${PORT}"
    echo "                      remote log: ${LOG_REMOTE}"
  ) &
  (( host_index++ )) || true   # prevent set -e from tripping on counter
done

wait

echo ""
echo "================================================================="
echo " Deployment finished — $(date)"
echo " Check ${LOG_FILE} for any errors."
echo "================================================================="