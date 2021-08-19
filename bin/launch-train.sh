#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
export PROJECT_DIR="$PWD"

ARGS=""
default_gpu='v100'
number_of_gpus=1
echo "options :"
while [ $# -gt 0 ]
do
    unset OPTIND
    unset OPTARG
    while getopts :g:n:j:  options
    do
    case $options in
        g)
          echo "-Selected GPU is: $OPTARG"
          default_gpu="$OPTARG"
          ;;
        n)
          echo "Number of GPUs=$OPTARG"
          number_of_gpus="$OPTARG"
          ;;
        j)
          echo "Job name is: $OPTARG"
          job_name=$OPTARG
          ;;
        \?)
          echo "Invalid option: -$OPTARG"
          exit 1
          ;;
        :)
          echo "Option -$OPTARG requires an argument."
          exit 1
          ;;
      esac
   done
   shift $((OPTIND-1))
   ARGS="${ARGS} $1"
   shift
done
if test -z "$job_name"
then
      args_array=($ARGS)
      job_name=${args_array[0]}
else
      echo "\$var is NOT empty"
fi
echo "ARGS :$ARGS"
echo "$@"
echo "--job-name $job_name --gres=gpu:$default_gpu:$number_of_gpus python run.py" $ARGS

export HYDRA_FULL_ERROR=1

# launch the training job
sbatch --job-name "$job_name" --gres "gpu:$default_gpu:$number_of_gpus" \
    "$PROJECT_DIR"/bin/train.sbatch "$PROJECT_DIR"/hello_world.py $ARGS