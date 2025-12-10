#!/bin/bash
#SBATCH --job-name=run_lr_rfecv
#SBATCH --output=run_lr_rfecv.out
#SBATCH --error=run_lr_rfecv.err
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem=20G
#SBATCH --time=96:00:00
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -w gnode063

# --- Environment setup ---
source ~/.bashrc
conda activate pyg

python train_lr_rfecv.py \
            --groups base bert_surprisal character_arcs emotional \
                     gc_academic gc_basic gc_char_diversity gc_concreteness gc_dialogue \
                     gc_discourse gc_narrative gc_polarity gc_pos gc_pronouns \
                     gc_punctuation gc_readability gc_syntax gc_temporal \
                     ngram ngram_surprisal plot_shifts rst structure surprisal \
            --oversample \
            --undersample \
            --undersample_method cluster_random \
            --undersample_clusters 20 \
            --prefer_gc_overlap \
            --output /scratch/ishaan.karan/lr_rfecv.pkl
