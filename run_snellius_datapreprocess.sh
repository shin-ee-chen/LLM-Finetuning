#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=data_preprocess-test
#SBATCH -e bash_output/data_preprocess-%j.err
#SBATCH -o bash_output/data_preprocess-%j.out
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00

module load 2023
module load Miniconda3/23.5.2-0
module load CUDA/12.1.1

cd LLM-Finetuning

python data_preprocess.py