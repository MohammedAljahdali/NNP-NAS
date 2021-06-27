#!/bin/bash --login
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J test_pruning
#SBATCH -o test_pruning.%J.out
#SBATCH -e test_pruning.%J.err
#SBATCH --time=01:30:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --constraint=[v100]
#SBATCH --output=logs/slurm/%x-slurm.out
#SBATCH --error=logs/slurm/%x-slurm.err

# entire script fails if single command fails
set -e

# activate the conda environment
module purge
conda activate sb-env

# use srun to launch NVDashboard server in order to reserve a port
# srun --resv-ports=1 ./bin/launch-nvdashboard-server.srun &
# NVDASHBOARD_PID=$!

mkdir /tmp/pruning_data

# launch the training script
python "$@"

rm -r /tmp/pruning_data

# shutdown the NVDashboard server
# kill $NVDASHBOARD_PID
