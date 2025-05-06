#!/bin/bash
#SBATCH --job-name=onnx_wtsquant
#SBATCH --output=log_wtsquant.out
#SBATCH --error=log_wtsquant.err
#SBATCH --partition=gpu    
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=16G

# Activate environment
source /home/njuttu_umass_edu/venvs/torch_env/bin/activate

# Navigate to working directory
cd /home/njuttu_umass_edu/on-DeviceNLP-2/Baseline_to_ONNX/Weights_Quantized

# Step 1: Export to quantized ONNX with shared embedding quantization
#python convert.py Helsinki-NLP/opus-mt-en-de -o ./quantWts_onnx

# Step 2: Evaluate BLEU score
python bleu_wmt.py --model-path ./quantWts_onnx/Helsinki-NLP_opus-mt-en-de