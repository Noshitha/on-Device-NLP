#!/bin/bash
#SBATCH --job-name=onnx_noquant
#SBATCH --output=logs/log_noquant_eval.out
#SBATCH --error=logs/log_noquant_eval.err
#SBATCH --partition=gpu    
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=16G

# 1) activate your env
source /home/njuttu_umass_edu/venvs/torch_env/bin/activate

# 2) go to the folder that contains convert.py, bleu_wmt.py, core/, etc.
cd /home/njuttu_umass_edu/on-DeviceNLP/Baseline_to_ONNX/Onnx_dynamic_Quant

# 3) make sure Python can see the 'core/' package
export PYTHONPATH=$(pwd):$(pwd)/..:$PYTHONPATH

echo "=== Converting model to ONNX (no quantize) ==="
python convert.py \
  Helsinki-NLP/opus-mt-fr-en \
  -o /home/njuttu_umass_edu/on-DeviceNLP/outs/no_quantize\
  --no-quantize
