#!/bin/bash
#SBATCH --job-name=onnx_noquant
#SBATCH --output=log_noquant.out
#SBATCH --error=log_noquant.err
#SBATCH --partition=gpu    
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=16G

source /home/njuttu_umass_edu/venvs/torch_env/bin/activate
cd /home/njuttu_umass_edu/on-DeviceNLP-2/Baseline_to_ONNX/Onnx_dynamic_Quant

python convert.py Helsinki-NLP/opus-mt-en-de -o ../outs_ONNX --no-quantize

# Evaluate BLEU score using the exported model
python Blue_wmt.py --model-path ../outs_ONNX/Helsinki-NLP_opus-mt-en-de
