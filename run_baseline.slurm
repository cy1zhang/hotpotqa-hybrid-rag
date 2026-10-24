#!/bin/bash
# Purdue Scholar GPU settings; adjust -A, -p and --constraint for your cluster.
#SBATCH -J Baseline_Sequential
#SBATCH -A gpu
#SBATCH -p gpu
#SBATCH --constraint=J
#SBATCH --output=logs/res_base_seq.txt
#SBATCH --error=logs/err_base_seq.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

export HF_HOME="${HF_HOME:-$PWD/hf_cache}"

mkdir -p logs

module load anaconda/2024.02-py311
conda activate aiForge

python -u 2_eval_baseline_rag.py