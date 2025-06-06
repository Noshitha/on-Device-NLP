#!/bin/bash
#SBATCH --job-name=onnx_prune_both
#SBATCH --output=logs/log_prune_both.out
#SBATCH --error=logs/log_prune_both.err
#SBATCH --partition=gpu    
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=16G

# 1) Activate your conda/venv environment
source /home/njuttu_umass_edu/venvs/torch_env/bin/activate

# 2) CD into the folder containing convert.py, prune_shared_embeddings.py, core/, etc.
cd /home/njuttu_umass_edu/on-DeviceNLP/Baseline_to_ONNX/Onnx_dynamic_Quant_pruned

# 3) Ensure Python can see the 'core/' package
export PYTHONPATH=$(pwd):$(pwd)/..:$PYTHONPATH

SRC="/home/njuttu_umass_edu/on-DeviceNLP/outs/no_quantize/Helsinki-NLP_opus-mt-fr-en"
DEST="/home/njuttu_umass_edu/on-DeviceNLP/outs/pruning/Helsinki-NLP_opus-mt-fr-en"

# Copy everything from the no_quantize output into a new pruning folder
mkdir -p "$DEST"
cp -a "$SRC/." "$DEST/"

echo "=== Pruning the encoder ==="
python prune_shared_embedding.py \
  -i "$DEST/encoder.onnx" \
  -o "$DEST/encoder.pruned.onnx" \
  -t 20000000

mv "$DEST/encoder.onnx"       "$DEST/encoder.orig.onnx"
mv "$DEST/encoder.pruned.onnx" "$DEST/encoder.onnx"
rm  "$DEST/encoder.orig.onnx"

echo
echo "=== Directory contents after encoder pruning ==="
ls -lh "$DEST"
echo

echo "=== Pruning the decoder ==="
python prune_shared_embedding.py \
  -i "$DEST/decoder.onnx" \
  -o "$DEST/decoder.pruned.onnx" \
  -t 20000000

mv "$DEST/decoder.onnx"       "$DEST/decoder.orig.onnx"
mv "$DEST/decoder.pruned.onnx" "$DEST/decoder.onnx"
rm  "$DEST/decoder.orig.onnx"

echo
echo "=== Directory contents after decoder pruning ==="
ls -lh "$DEST"
echo

echo "=== Sample test on Pruned encoder+decoder ==="
python test_pruning.py  # make sure this script still points to the same MODEL_DIR

echo "=== BLEU on pruned encoder+decoder ==="
python bleu_wmt.py \
  --model-path "$DEST" \
  --output-dir "$DEST/eval_after_prune" \
  --split "validation[:100]"
