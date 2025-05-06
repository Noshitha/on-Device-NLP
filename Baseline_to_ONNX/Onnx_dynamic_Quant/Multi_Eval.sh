#!/bin/bash
#SBATCH --job-name=bleu_multi
#SBATCH --output=log_bleu_multi.out
#SBATCH --error=log_bleu_multi.err
#SBATCH --partition=gpu    
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G

# Activate Python environment
source /home/njuttu_umass_edu/venvs/torch_env/bin/activate

# Path to evaluation script
BLEU_SCRIPT="/home/njuttu_umass_edu/on-DeviceNLP-2/Baseline_to_ONNX/Weights_Quantized/bleu_wmt.py"

# -------------------------
# Baseline ONNX model BLEU
# -------------------------
BASELINE_MODEL="/home/njuttu_umass_edu/on-DeviceNLP-2/Baseline_to_ONNX/outs_ONNX/Helsinki-NLP_opus-mt-en-de"

python "$BLEU_SCRIPT" --model-path "$BASELINE_MODEL" --split "validation[:50]" --output-dir "$BASELINE_MODEL/bleu_eval_50"
python "$BLEU_SCRIPT" --model-path "$BASELINE_MODEL" --split "validation[:100]" --output-dir "$BASELINE_MODEL/bleu_eval_100"
python "$BLEU_SCRIPT" --model-path "$BASELINE_MODEL" --split "validation[:500]" --output-dir "$BASELINE_MODEL/bleu_eval_500"
python "$BLEU_SCRIPT" --model-path "$BASELINE_MODEL" --split "validation[:1000]" --output-dir "$BASELINE_MODEL/bleu_eval_1000"

# ----------------------------------------
# Quantized ONNX model BLEU (Dynamic Quant)
# ----------------------------------------
QUANT_MODEL="/home/njuttu_umass_edu/on-DeviceNLP-2/Baseline_to_ONNX/outs_ONNX_quantized/Helsinki-NLP_opus-mt-en-de"

python "$BLEU_SCRIPT" --model-path "$QUANT_MODEL" --split "validation[:50]" --output-dir "$QUANT_MODEL/bleu_eval_50"
python "$BLEU_SCRIPT" --model-path "$QUANT_MODEL" --split "validation[:100]" --output-dir "$QUANT_MODEL/bleu_eval_100"
python "$BLEU_SCRIPT" --model-path "$QUANT_MODEL" --split "validation[:500]" --output-dir "$QUANT_MODEL/bleu_eval_500"
python "$BLEU_SCRIPT" --model-path "$QUANT_MODEL" --split "validation[:1000]" --output-dir "$QUANT_MODEL/bleu_eval_1000"
