#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=train.out
#SBATCH --error=train.err
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH --time=96:00:00
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -w gnode068


# --- Environment setup ---
source ~/.bashrc
conda activate pyg

python train.py --mode train   
