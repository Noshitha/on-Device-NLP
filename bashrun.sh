#!/bin/bash

#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --mem=80GB
#SBATCH --time=48:00:00
module load miniconda/22.11.1-1
conda init
conda activate adobeSudhanshuEnv 
# Construct the filename based on the passed argument and run the corresponding Python script
SCRIPT_NAME="/work/pi_wenlongzhao_umass_edu/3/Sudhanshu/on-DeviceNLP/translation/baseline/baseline_test.py"

if [ -f "$SCRIPT_NAME" ]; then
    echo "Running script: $SCRIPT_NAME"
    python3 "$SCRIPT_NAME"
else
    echo "Script not found: $SCRIPT_NAME"
    exit 1
fi