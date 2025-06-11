#!/bin/bash
#SBATCH --job-name=onnx_eval_unpruned
#SBATCH --output=/home/njuttu_umass_edu/on-DeviceNLP/VocabRed_to_ONNX/logs/eval_unpruned.out
#SBATCH --error=/home/njuttu_umass_edu/on-DeviceNLP/VocabRed_to_ONNX/logs/eval_unpruned.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=16G

# activate your venv
source /home/njuttu_umass_edu/venvs/torch_env/bin/activate

# go to the directory containing evaluation.py
cd /home/njuttu_umass_edu/on-DeviceNLP/VocabRed_to_ONNX/Onnx_dynamic_Quant

echo "=== Sample test on ONNX encoder+decoder ==="
python /home/njuttu_umass_edu/on-DeviceNLP/VocabRed_to_ONNX/Onnx_dynamic_Quant/test_sample.py

echo "=== All metrics Evaluation ==="
python /home/njuttu_umass_edu/on-DeviceNLP/VocabRed_to_ONNX/Onnx_dynamic_Quant/evaluation.py \
  --model-dir /home/njuttu_umass_edu/on-DeviceNLP/outs/Vocab_Red_NoQuant/checkpoint-4924000 \
  --onnx-dir  /home/njuttu_umass_edu/on-DeviceNLP/outs/Vocab_Red_NoQuant/checkpoint-4924000 \
  --split "validation[:100]"
