#!/bin/bash
# Run one Go cost measurement on a CPU node. No GPU is requested.
#
#   sbatch slurm/cpu.sbatch -cost-vs-d
#
# The flags are those of fhe/heoft-fhe, listed in fhe/README.md. Set the
# partition and account for your cluster. Bootstrapping keys need tens of GB.
#SBATCH --job-name=heoft-fhe
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/%x_%j.out
#SBATCH --error=outputs/slurm/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}/fhe"
go build -o heoft-fhe .
exec srun ./heoft-fhe "$@"
