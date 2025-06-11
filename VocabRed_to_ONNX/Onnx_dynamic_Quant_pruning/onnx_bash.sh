#!/bin/bash
#SBATCH --job-name=onnx_vocabred
#SBATCH --output=/home/njuttu_umass_edu/on-DeviceNLP/VocabRed_to_ONNX/logs/pruning_eval.out
#SBATCH --error=/home/njuttu_umass_edu/on-DeviceNLP/VocabRed_to_ONNX/logs/pruning_eval.err
#SBATCH --partition=gpu    
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=16G

# 1) Activate your virtual environment (containing torch, onnx, etc.)
source /home/njuttu_umass_edu/venvs/torch_env/bin/activate

# 2) Change into the directory where prune scripts live
cd /home/njuttu_umass_edu/on-DeviceNLP/VocabRed_to_ONNX

# 3) Paths
# ORIG_DIR=/home/njuttu_umass_edu/on-DeviceNLP/outs/Vocab_Red_NoQuant/checkpoint-4924000
# PRUNED_DIR=/home/njuttu_umass_edu/on-DeviceNLP/outs/Vocab_Red_NoQuant/checkpoint-4924000_pruned

# ### UPDATED: point to the `prune_by_name.py` script you created earlier
# PRUNE_SCRIPT=/home/njuttu_umass_edu/on-DeviceNLP/VocabRed_to_ONNX/Onnx_dynamic_Quant_pruning/prune_name.py

# ### UPDATED: point to the `list_initializers.py` helper you created earlier
# LIST_INITS=/home/njuttu_umass_edu/on-DeviceNLP/VocabRed_to_ONNX/Onnx_dynamic_Quant_pruning/list_initalizers.py

# # 4) Copy the entire original directory to a new “pruned” folder
# echo "Copying original folder to pruned folder:"
# cp -r "$ORIG_DIR" "$PRUNED_DIR"

# # 5) List initializers in encoder.onnx (sanity check)
# echo "Listing initializers in encoder.onnx:"
# python "$LIST_INITS" "$PRUNED_DIR/encoder.onnx"

# # 6) Prune encoder.embed_tokens.weight by name
# echo "Pruning encoder.onnx (encoder.embed_tokens.weight) ..."
# python "$PRUNE_SCRIPT" \
#   -i "$PRUNED_DIR/encoder.onnx" \
#   -o "$PRUNED_DIR/encoder_pruned.onnx" \
#   -n encoder.embed_tokens.weight
# mv "$PRUNED_DIR/encoder_pruned.onnx" "$PRUNED_DIR/encoder.onnx"

# # 7) List initializers in decoder.onnx (sanity check)
# echo "Listing initializers in decoder.onnx:"
# python "$LIST_INITS" "$PRUNED_DIR/decoder.onnx"

# # 8) Prune decoder.embed_tokens.weight by name
# echo "Pruning decoder.onnx (decoder.embed_tokens.weight) ..."
# python "$PRUNE_SCRIPT" \
#   -i "$PRUNED_DIR/decoder.onnx" \
#   -o "$PRUNED_DIR/decoder_pruned.onnx" \
#   -n decoder.embed_tokens.weight
# mv "$PRUNED_DIR/decoder_pruned.onnx" "$PRUNED_DIR/decoder.onnx"

# # 9) Show final file sizes to confirm pruning
# echo "Pruned file sizes:"
# ls -lh "$PRUNED_DIR/encoder.onnx" "$PRUNED_DIR/decoder.onnx"

# echo "All done. Your pruned ONNX files now live in: $PRUNED_DIR"


echo "=== Sample test on Pruned encoder+decoder ==="
python /home/njuttu_umass_edu/on-DeviceNLP/VocabRed_to_ONNX/Onnx_dynamic_Quant_pruning/test_sample.py

echo "=== All metrics Evaluation ==="
python /home/njuttu_umass_edu/on-DeviceNLP/VocabRed_to_ONNX/Onnx_dynamic_Quant_pruning/evaluation.py \
  --model-dir /home/njuttu_umass_edu/on-DeviceNLP/outs/Vocab_Red_NoQuant/checkpoint-4924000_pruned \
  --onnx-dir  /home/njuttu_umass_edu/on-DeviceNLP/outs/Vocab_Red_NoQuant/checkpoint-4924000_pruned \
  --split "validation[:100]" \
  --output-dir /home/njuttu_umass_edu/on-DeviceNLP/outs/Vocab_Red_NoQuant/checkpoint-4924000_pruned/translations
