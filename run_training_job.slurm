#!/bin/bash
#SBATCH -J JupyterServer
#SBATCH -t 3:00:00
#SBATCH -n 1
#SBATCH --mem=120G
#SBATCH --partition=gpu-a100 --gres=gpu:1
#SBATCH --output=./slurm_out/%j.out

source ~/miniconda3/etc/profile.d/conda.sh && \
conda activate triagerx && \
python notebook/training.py