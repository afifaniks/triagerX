#!/bin/bash
#SBATCH -J triagerx
#SBATCH -t 0-06:00:00
#SBATCH -n 1
#SBATCH --mem=80G
#SBATCH --partition=disa --gres=gpu:1
#SBATCH --output=/home/mdafifal.mamun/notebooks/triagerX/slurm_out/%j.out

source ~/miniconda3/etc/profile.d/conda.sh && \
conda activate triagerx && \
python /home/mdafifal.mamun/notebooks/triagerX/util/issue_summarizer.py