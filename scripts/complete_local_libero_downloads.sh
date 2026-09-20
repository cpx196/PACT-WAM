#!/usr/bin/env bash
# Finish all files required by the local FastWAM LIBERO evaluator.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/local_libero_env.sh"

# The server is admitted to the campus network before this script is run, so
# downloads are direct and do not depend on an SSH reverse proxy.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
FASTWAM_DOWNLOAD_WORKERS="${FASTWAM_DOWNLOAD_WORKERS:-4}"

T5_OUTPUT="${DIFFSYNTH_MODEL_BASE_PATH}/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors"
T5_URL='https://www.modelscope.cn/api/v1/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/repo?Revision=master&FilePath=models_t5_umt5-xxl-enc-bf16.safetensors'
LIBERO_CKPT="${FASTWAM_ROOT}/checkpoints/fastwam_release/libero_uncond_2cam224.pt"
LIBERO_URL='https://hf-mirror.com/yuanty/fastwam/resolve/main/libero_uncond_2cam224.pt?download=true'

"${FASTWAM_PYTHON}" "${SCRIPT_DIR}/download_hf_file_parallel.py" \
  "${T5_URL}" "${T5_OUTPUT}" \
  --size 11361845432 --chunk-mb 1 \
  --workers "${FASTWAM_DOWNLOAD_WORKERS}" --retries 100

while ! "${FASTWAM_PYTHON}" "${SCRIPT_DIR}/download_hf_file_parallel.py" \
  "${LIBERO_URL}" "${LIBERO_CKPT}" \
  --size 12041735140 --chunk-mb 8 \
  --workers "${FASTWAM_DOWNLOAD_WORKERS}" --retries 100; do
  echo "Checkpoint download stopped after repeated network errors; resuming in 10 seconds." >&2
  sleep 10
done

"${FASTWAM_PYTHON}" - <<'PY'
import json
import zipfile
from pathlib import Path

from fastwam.models.wan22.helpers.io import hash_model_file

root = Path(__import__("os").environ["FASTWAM_ROOT"])
wan = Path(__import__("os").environ["DIFFSYNTH_MODEL_BASE_PATH"])
common = wan / "DiffSynth-Studio/Wan-Series-Converted-Safetensors"
vae = common / "Wan2.2_VAE.safetensors"
t5 = common / "models_t5_umt5-xxl-enc-bf16.safetensors"
ckpt = root / "checkpoints/fastwam_release/libero_uncond_2cam224.pt"
stats = root / "checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json"
tokenizer = wan / "Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"

assert vae.stat().st_size == 1409401152, vae
assert t5.stat().st_size == 11361845432, t5
assert ckpt.stat().st_size == 12041735140, ckpt
assert hash_model_file(str(vae)) == "e1de6c02cdac79f8b739f4d3698cd216"
assert hash_model_file(str(t5)) == "9c8818c2cbea55eca56c7b447df170da"
assert zipfile.is_zipfile(ckpt), ckpt
json.loads(stats.read_text(encoding="utf-8"))
for name in ("special_tokens_map.json", "spiece.model", "tokenizer.json", "tokenizer_config.json"):
    assert (tokenizer / name).is_file(), tokenizer / name
print("All FastWAM LIBERO model files passed local validation.")
PY
