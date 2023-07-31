#!/bin/bash
#SBATCH -e bash_output/llama2-lora-%j.err
#SBATCH -o bash_output/llama2-lora-%j.out
#SBATCH -J  PEFT
#SBATCH --nodelist=gpu08
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00

CUDA_VISIBLE_DEVICES=1 python finetune.py \
--base_model '/data03/irlab_share/Llama-2-7b-hf' \
--data_path './datasets/alpaca_data_cleaned.json' \
--output_dir './lora-alpaca' \
--batch_size 128 \
--micro_batch_size 32