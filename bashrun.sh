#!/bin/bash

#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --mem=80GB
#SBATCH --time=48:00:00

source /home/njuttu_umass_edu/venvs/torch_env/bin/activate

python 