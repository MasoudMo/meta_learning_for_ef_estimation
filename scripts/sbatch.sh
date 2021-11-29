#!/bin/sh

#SBATCH --account=def-dsuth
#SBATCH --gres=gpu:1              # Number of GPU(s) per node
#SBATCH --cpus-per-task=8         # CPU cores/threads
#SBATCH --mem=64000M               # memory per node
#SBATCH --time=2-00:00            # time (DD-HH:MM)
#SBATCH --output=slurm_output/slurm-%j.out

source $LMOD_PKG/init/zsh
source ENV/bin/activate
./scripts/run.sh "$@"

