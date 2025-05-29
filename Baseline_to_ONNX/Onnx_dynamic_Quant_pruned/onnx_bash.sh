#!/bin/bash
#SBATCH --job-name=onnx_noquant
#SBATCH --output=logs/log_prune_eval.out
#SBATCH --error=logs/log_prune_eval.err
#SBATCH --partition=gpu    
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=16G

# 1) activate your env
source /home/njuttu_umass_edu/venvs/torch_env/bin/activate

# 2) go to the folder that contains convert.py, bleu_wmt.py, core/, etc.
cd /home/njuttu_umass_edu/on-DeviceNLP/Baseline_to_ONNX/Onnx_dynamic_Quant_pruned

# 3) make sure Python can see the 'core/' package
export PYTHONPATH=$(pwd):$(pwd)/..:$PYTHONPATH

SRC="/home/njuttu_umass_edu/on-DeviceNLP/outs/no_quantize"
DEST="/home/njuttu_umass_edu/on-DeviceNLP/outs/pruning"

# create the destination directory (won't error if it already exists)
mkdir -p "$DEST"

# copy all files and subdirectories, preserving permissions/timestamps
cp -a "$SRC/." "$DEST/"

echo "=== Pruning only the decoder ==="
python /home/njuttu_umass_edu/on-DeviceNLP/Baseline_to_ONNX/Onnx_dynamic_Quant_pruned/prune_shared_embedding.py \
  -i /home/njuttu_umass_edu/on-DeviceNLP/outs/pruning/Helsinki-NLP_opus-mt-fr-en/decoder.onnx \
  -o /home/njuttu_umass_edu/on-DeviceNLP/outs/pruning/Helsinki-NLP_opus-mt-fr-en/decoder.pruned.onnx \
  -t 20000000

echo "=== Swapping in pruned decoder ==="
mv /home/njuttu_umass_edu/on-DeviceNLP/outs/pruning/Helsinki-NLP_opus-mt-fr-en/decoder.onnx \
   /home/njuttu_umass_edu/on-DeviceNLP/outs/pruning/Helsinki-NLP_opus-mt-fr-en/decoder.orig.onnx
mv /home/njuttu_umass_edu/on-DeviceNLP/outs/pruning/Helsinki-NLP_opus-mt-fr-en/decoder.pruned.onnx \
   /home/njuttu_umass_edu/on-DeviceNLP/outs/pruning/Helsinki-NLP_opus-mt-fr-en/decoder.onnx

echo "=== Sample test on Pruned decoder ==="
python /home/njuttu_umass_edu/on-DeviceNLP-2/Baseline_to_ONNX/Onnx_dynamic_Quant/test_pruning.py


echo "=== BLEU on pruned decoder ==="
python bleu_wmt.py \
  --model-path /home/njuttu_umass_edu/on-DeviceNLP/outs/pruning/Helsinki-NLP_opus-mt-fr-en\
  --output-dir /home/njuttu_umass_edu/on-DeviceNLP/outs/pruning/Helsinki-NLP_opus-mt-fr-en/eval_after_prune \
  --split "validation[:100]"
