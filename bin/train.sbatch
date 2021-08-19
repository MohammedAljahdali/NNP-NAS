#!/bin/bash --login
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/slurm/%j-%x-slurm.out
#SBATCH --error=logs/slurm/%j-%x-slurm.err

# entire script fails if single command fails
set -e

# activate the conda environment
module purge
conda activate nnp-nas-2

# use srun to launch NVDashboard server in order to reserve a port
# srun --resv-ports=1 ./bin/launch-nvdashboard-server.srun &
# NVDASHBOARD_PID=$!

mkdir /tmp/pruning_data
# mkdir /tmp/pruning_runs

# launch the training script
python "$@"

# cp -r /tmp/pruning_runs /home/aljahdmk/project/shrinkbench/runs
rm -r /tmp/pruning_data
# rm -r /tmp/pruning_runs

# shutdown the NVDashboard server
# kill $NVDASHBOARD_PID
