#!/bin/bash

#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --mem=80GB
#SBATCH --time=48:00:00

source /home/njuttu_umass_edu/venvs/torch_env/bin/activate

#python /home/njuttu_umass_edu/on-DeviceNLP-2/Embedding_Weights_Quantization/convert.py Helsinki-NLP/opus-mt-en-de #create the Embedding weights model

python /home/njuttu_umass_edu/on-DeviceNLP-2/Embedding_Weights_Quantization/bleu_wmt.py  #evaluating the model