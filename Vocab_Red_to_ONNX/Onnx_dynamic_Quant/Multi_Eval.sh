#!/bin/bash
#SBATCH --job-name=bleu_multi
#SBATCH --output=/home/njuttu_umass_edu/on-DeviceNLP-2/Vocab_Red_to_ONNX/logs/log_bleu_multi.out
#SBATCH --error=/home/njuttu_umass_edu/on-DeviceNLP-2/Vocab_Red_to_ONNX/logs/log_bleu_multi.err
#SBATCH --partition=gpu    
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G

# Activate Python environment
source /home/njuttu_umass_edu/venvs/torch_env/bin/activate

# Path to evaluation script
BLEU_SCRIPT="/home/njuttu_umass_edu/on-DeviceNLP-2/Vocab_Red_to_ONNX/Onnx_dynamic_Quant/Bleu_WMT.py"

# -------------------------
# Vocab reduced ONNX model BLEU
# -------------------------
VOCABRED_MODEL="/home/njuttu_umass_edu/on-DeviceNLP-2/Vocab_Red_to_ONNX/Onnx_dynamic_Quant/vocab_red_onnx_output/en-fr-8000-1000000-finetuned"

python "$BLEU_SCRIPT" --model-path "$VOCABRED_MODEL" --split "validation[:50]" --src-lang "en" --tgt-lang "fr" --output-dir "$VOCABRED_MODEL/bleu_eval_50"
python "$BLEU_SCRIPT" --model-path "$VOCABRED_MODEL" --split "validation[:100]" --src-lang "en" --tgt-lang "fr" --output-dir "$VOCABRED_MODEL/bleu_eval_100"
python "$BLEU_SCRIPT" --model-path "$VOCABRED_MODEL" --split "validation[:500]" --src-lang "en" --tgt-lang "fr" --output-dir "$VOCABRED_MODEL/bleu_eval_500"
python "$BLEU_SCRIPT" --model-path "$VOCABRED_MODEL" --split "validation[:1000]" --src-lang "en" --tgt-lang "fr" --output-dir "$VOCABRED_MODEL/bleu_eval_1000"

# ----------------------------------------
# Quantized Vocab reduced ONNX model BLEU (Dynamic Quant)
# ----------------------------------------
QUANT_MODEL="/home/njuttu_umass_edu/on-DeviceNLP-2/Vocab_Red_to_ONNX/Onnx_dynamic_Quant/vocab_red_onnx_output_quant/en-fr-8000-1000000-finetuned"

python "$BLEU_SCRIPT" --model-path "$QUANT_MODEL" --split "validation[:50]" --src-lang "en" --tgt-lang "fr" --output-dir "$QUANT_MODEL/bleu_eval_50"
python "$BLEU_SCRIPT" --model-path "$QUANT_MODEL" --split "validation[:100]" --src-lang "en" --tgt-lang "fr" --output-dir "$QUANT_MODEL/bleu_eval_100"
python "$BLEU_SCRIPT" --model-path "$QUANT_MODEL" --split "validation[:500]" --src-lang "en" --tgt-lang "fr" --output-dir "$QUANT_MODEL/bleu_eval_500"
python "$BLEU_SCRIPT" --model-path "$QUANT_MODEL" --split "validation[:1000]" --src-lang "en" --tgt-lang "fr" --output-dir "$QUANT_MODEL/bleu_eval_1000"
