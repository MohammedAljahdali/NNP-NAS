#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
export PROJECT_DIR="$PWD"

# launch the training job
sbatch --job-name "$JOB_NAME" \
    "$PROJECT_DIR"/bin/train.sbatch "$PROJECT_DIR"/run.py "$@"

