#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=finetune_llama3_lora-test
#SBATCH -e bash_output/llama3-lora-%j.err
#SBATCH -o bash_output/llama3-lora-%j.out
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00

module load 2022
module load Anaconda3/2022.05
source activate llm2023

cd LLM-Finetuning
python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('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')"

python finetune.py \
--base_model 'meta-llama/Meta-Llama-3-8B-Instruct' \
--data_path './datasets/rag_toy.json' \
--output_dir './lora-rag_toy' \
--batch_size 1 \
--micro_batch_size 1 \
--wandb_project llm-finetuning \
--wandb_run_name default-test \
--prompt_template_name kdd_baseline \
--val_set_size 1