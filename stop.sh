#!/bin/bash

HOSTS=(
  ip-10-0-130-168
  ip-10-0-130-172
  ip-10-0-130-183
  ip-10-0-132-247
  ip-10-0-133-47
  ip-10-0-135-197
  ip-10-0-144-104
  ip-10-0-144-22
  ip-10-0-150-146
  ip-10-0-153-3
  ip-10-0-153-87
  ip-10-0-154-148
  ip-10-0-157-253
  ip-10-0-173-237
  ip-10-0-182-181
  ip-10-0-192-124
  ip-10-0-208-224
  ip-10-0-213-235
  ip-10-0-215-23
  ip-10-0-221-137
  ip-10-0-224-46
  ip-10-0-227-25
  ip-10-0-228-101
  ip-10-0-236-153
  ip-10-0-249-56
  ip-10-0-253-185
  ip-10-0-255-251
)

SSH="ssh -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ConnectTimeout=10"

echo "================================================================="
echo " Killing vllm tmux sessions — $(date)"
echo " Hosts : ${#HOSTS[@]}"
echo "================================================================="

for HOST in "${HOSTS[@]}"; do
  (
    SESSION="vllm_${HOST}"
    CHECK=$($SSH "$HOST" "echo ok" 2>/dev/null)
    if [[ "$CHECK" != "ok" ]]; then
      echo "[$(date '+%H:%M:%S')] ✗ ${HOST} — unreachable, skipping"
      exit 1
    fi

    $SSH "$HOST" "tmux kill-session -t '${SESSION}' 2>/dev/null" 2>/dev/null
    echo "[$(date '+%H:%M:%S')] ✓ ${HOST} — session '${SESSION}' killed"
  ) &
done

wait

echo ""
echo "================================================================="
echo " Done — $(date)"
echo "================================================================="