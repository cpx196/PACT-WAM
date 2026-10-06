#!/usr/bin/env bash
set -euo pipefail
cd /data/pxchen/PACT-WAM
root=artifacts/libero10_predictor_energy_ranking_pilot100_20260930
mkdir -p "$root"
while true; do
  candidate=""
  for gpu in 0 1 2 3 4 5 6 7; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
    if [ "$used" -le 512 ]; then
      candidate="$gpu"
      break
    fi
  done
  if [ -n "$candidate" ]; then
    stable=1
    for _ in 1 2 3 4 5 6; do
      sleep 5
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$candidate" | tr -d ' ')
      if [ "$used" -gt 512 ]; then stable=0; break; fi
    done
    if [ "$stable" -eq 1 ]; then
      echo "$(date '+%F %T') launching on GPU $candidate" >> "$root/wait.log"
      CUDA_VISIBLE_DEVICES="$candidate" exec bash scripts/run_predictor_energy_ranking_pilot100.sh
    fi
  fi
  sleep 20
done
