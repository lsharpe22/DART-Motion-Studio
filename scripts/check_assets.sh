#!/usr/bin/env bash
set -euo pipefail
DART_HOME="${DART_HOME:-/content/DART}"
DART_PYTHON="${DART_PYTHON:-/content/dart-env/bin/python}"
fail=0
check() { if [[ -f "$DART_HOME/$1" ]]; then printf 'OK      %s\n' "$1"; else printf 'MISSING %s\n' "$1"; fail=1; fi; }
echo "DART_HOME=$DART_HOME"
echo "DART_PYTHON=$DART_PYTHON"
[[ -x "$DART_PYTHON" ]] || { echo 'ERROR: Python executable missing'; exit 1; }
for file in motion_service/backend/persistent_worker.py motion_service/backend/api_persistent.py mld/rollout_mld.py mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt mld_denoiser/mld_fps_clip_repeat_euler/args.yaml mvae/mvae_fps_clip/checkpoint_200000.pt mvae/mvae_fps_clip/args.yaml data/stand.pkl data/seq_data_zero_male/mean_std_h2_f8.pkl body_models/smplx/SMPLX_MALE.npz; do check "$file"; done
if [[ "$fail" -ne 0 ]]; then echo 'Missing required files. Consult REPRODUCIBILITY.md.'; exit 1; fi
(cd "$DART_HOME" && PYTHONPATH="$DART_HOME:$DART_HOME/motion_service/backend${PYTHONPATH:+:$PYTHONPATH}" "$DART_PYTHON" -c 'import sys, urllib.parse, torch, fastapi, uvicorn, mld.rollout_mld; print("Python:", sys.version.split()[0]); print("PyTorch:",torch.__version__); assert torch.cuda.is_available(), "CUDA GPU unavailable"; print("GPU:",torch.cuda.get_device_name(0)); print("DART and service imports OK")')
