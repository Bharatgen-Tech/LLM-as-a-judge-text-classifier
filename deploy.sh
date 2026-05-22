#!/bin/bash

# =============================================================================
# deploy_vllm.sh
# Concurrently deploys vllm on a list of remote hosts, two tmux panes each:
#   Pane 0 → GPUs 0,1,2,3  port 30200
#   Pane 1 → GPUs 4,5,6,7  port 30201
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/deployment_errors.log"

HOSTS=(
  ip-10-0-192-124
  ip-10-0-208-224 
  ip-10-0-236-153 
  ip-10-0-253-185
  ip-10-0-255-251
)

echo "================================================================="
echo " vllm concurrent deployment — $(date)"
echo " Hosts : ${#HOSTS[@]}"
echo " Log   : ${LOG_FILE}"
echo "================================================================="

for HOST in "${HOSTS[@]}"; do
  (
    SESSION="vllm_${HOST}"
    SSH="ssh -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=3"

    CMD0='source /fsxnew/sharvil.palvekar/benchmarks/ocr_books/scripts/OCR_w_dots/fabric/vllmenv/bin/activate; CUDA_VISIBLE_DEVICES=0,1,2,3 uv run vllm serve openai/gpt-oss-120b --tensor-parallel-size 4 --port 30200 --gpu-memory-utilization 0.90 --enable-prefix-caching --trust-remote-code --generation-config vllm --max-num-seqs 32 --max-num-batched-tokens 24576 --max-model-len 65536 --async-scheduling'

    CMD1='source /fsxnew/sharvil.palvekar/benchmarks/ocr_books/scripts/OCR_w_dots/fabric/vllmenv/bin/activate; CUDA_VISIBLE_DEVICES=4,5,6,7 uv run vllm serve openai/gpt-oss-120b --tensor-parallel-size 4 --port 30201 --gpu-memory-utilization 0.90 --enable-prefix-caching --trust-remote-code --generation-config vllm --max-num-seqs 32 --max-num-batched-tokens 24576 --max-model-len 65536 --async-scheduling'

    echo "[$(date '+%H:%M:%S')] Deploying → ${HOST}"

    # SSH connectivity check
    CHECK=$($SSH "$HOST" "echo ok" 2>/dev/null)
    if [[ "$CHECK" != "ok" ]]; then
      ERR=$($SSH "$HOST" "echo ok" 2>&1)
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEPLOYMENT ERROR | host=${HOST} pane=ssh-check | ${ERR}" | tee -a "$LOG_FILE"
      exit 1
    fi

    # Kill stale session
    $SSH "$HOST" "tmux kill-session -t '${SESSION}' 2>/dev/null; true" 2>/dev/null

    # ── Pane 0: GPUs 0-3 port 30200 ──────────────────────────────────────────
    ERR=$($SSH "$HOST" "tmux new-session -d -s '${SESSION}' -x 220 -y 50 \
      'bash -lc $(printf '%q' "${CMD0}") 2>&1 | tee ~/vllm_gpu0123_\$(date +%Y%m%d_%H%M%S).log'" 2>&1)
    if [[ $? -ne 0 ]]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEPLOYMENT ERROR | host=${HOST} pane=pane0(GPUs 0-3 :30200) | tmux new-session failed: ${ERR}" | tee -a "$LOG_FILE"
      exit 1
    fi

    # ── Pane 1: GPUs 4-7 port 30201 ──────────────────────────────────────────
    # Retry loop: wait up to 3s for window 0 to be ready before split-window.
    # Under concurrent SSH load the window isn't always ready immediately.
    ERR=$($SSH "$HOST" "
      for i in \$(seq 1 10); do
        tmux list-windows -t '${SESSION}' 2>/dev/null | grep -q '^0:' && break
        sleep 0.3
      done
      tmux split-window -t '${SESSION}:0' -v \
        'bash -lc $(printf '%q' "${CMD1}") 2>&1 | tee ~/vllm_gpu4567_\$(date +%Y%m%d_%H%M%S).log'
    " 2>&1)
    if [[ $? -ne 0 ]]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEPLOYMENT ERROR | host=${HOST} pane=pane1(GPUs 4-7 :30201) | tmux split-window failed: ${ERR}" | tee -a "$LOG_FILE"
      exit 1
    fi

    echo "[$(date '+%H:%M:%S')] ✓ ${HOST} — both panes started (session: ${SESSION})"
  ) &
done

wait

echo ""
echo "================================================================="
echo " Deployment finished — $(date)"
echo " Check ${LOG_FILE} for any errors."
echo "================================================================="