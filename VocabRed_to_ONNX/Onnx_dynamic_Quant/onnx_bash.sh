#!/bin/bash
#SBATCH --job-name=onnx_vocabred
#SBATCH --output=/home/njuttu_umass_edu/on-DeviceNLP/VocabRed_to_ONNX/logs/log_vocabred.out
#SBATCH --error=/home/njuttu_umass_edu/on-DeviceNLP/VocabRed_to_ONNX/logs/log_vocabred.err
#SBATCH --partition=gpu    
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=16G

source /home/njuttu_umass_edu/venvs/torch_env/bin/activate
cd /home/njuttu_umass_edu/on-DeviceNLP/VocabRed_to_ONNX

python /home/njuttu_umass_edu/on-DeviceNLP/VocabRed_to_ONNX/Onnx_dynamic_Quant/convert.py \
  --model-dir /home/njuttu_umass_edu/on-DeviceNLP/VocabRed_to_ONNX/checkpoint-4924000 \
  --tokenizer-dir /home/njuttu_umass_edu/on-DeviceNLP/VocabRed_to_ONNX/checkpoint-4924000 \
  -o /home/njuttu_umass_edu/on-DeviceNLP/outs/Vocab_Red_NoQuant \
  --no-quantize