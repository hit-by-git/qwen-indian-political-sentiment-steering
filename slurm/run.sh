#!/bin/bash
#SBATCH --job-name=qwen-steering
#SBATCH --partition=rtx6000-fullgpu
#SBATCH --gres=gpu:rtx_pro:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=22G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${MODEL_DIR:?Set MODEL_DIR to the staged model directory}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
df -h .
if command -v quota >/dev/null; then quota -s || true; fi
python src/gpu.py "$1" --model-dir "$MODEL_DIR" "${@:2}"
