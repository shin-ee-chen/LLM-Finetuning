#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=finetune_llama2_lora-test
#SBATCH -e bash_output/llama2-lora-%j.err
#SBATCH -o bash_output/llama2-lora-%j.out
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00

module load 2022
module load Anaconda3/2022.05
source activate llm2023

cd LLM-Finetuning

python finetune.py \
--base_model '/home/xchen/HELM-local-evaluation/models/huggingface/Llama-2-7b-hf' \
--data_path './datasets/alpaca_data_cleaned.json' \
--output_dir './lora-alpaca' \
--batch_size 128 \
--micro_batch_size 64 \
--wandb_project llm-finetuning \
--wandb_run_name default-test