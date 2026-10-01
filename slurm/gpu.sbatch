#!/bin/bash
# Run one Python experiment on a GPU node.
#
#   sbatch slurm/gpu.sbatch accuracy_text --tasks ag_news --seeds 42
#   sbatch slurm/gpu.sbatch accuracy_vision --datasets cifar10 --clients 20 --alpha 0.04 --max-test 10000
#
# Set the partition, account and GPU type for your cluster. The paper's runs
# used one NVIDIA A40 (48 GB) per job. If compute nodes have no internet, run
# experiments/prefetch.py once on a login node and keep HF_HUB_OFFLINE=1.
#SBATCH --job-name=heoft
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/%x_%j.out
#SBATCH --error=outputs/slurm/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
# Activate the environment that has `pip install -e .` here.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
exp="$1"; shift
exec srun python -u "experiments/${exp}.py" "$@"
