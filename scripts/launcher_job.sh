#!/usr/bin/env bash

#SBATCH -p <YOUR_GPU_PARTITION>
#SBATCH -t 0-01:00:00
#SBATCH --gres gpu:1
#SBATCH -o "<YOUR_SLURM_LOG_DIR>/%x-%A-%a.out"

echo "$SLURM_JOB_ID" > "$SLURM_JOB_ID"

eval "$(conda shell.bash hook)"

conda activate <YOUR_CONDA_ENV>

echo "Hello World"

nvidia-smi

printf "Executing: %s\n" "$EXEC_COMMAND"
bash -c "$EXEC_COMMAND"

echo Finished
