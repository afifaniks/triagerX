#!/bin/bash
#SBATCH -J triagerx
#SBATCH -t 6:00:00
#SBATCH -n 1
#SBATCH --mem=50G
#SBATCH --partition=gpu-a100 --gres=gpu:1
#SBATCH --output=%j.out

source ~/miniconda3/etc/profile.d/conda.sh && \
conda activate trx && \
python deberta_cv1.py