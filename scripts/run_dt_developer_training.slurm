#!/bin/bash
#SBATCH -J deeptriagerx
#SBATCH -t 0-24:00:00
#SBATCH -n 1
#SBATCH --mem=80G
#SBATCH --partition=gpu-a100 --gres=gpu:1
#SBATCH --output=/home/mdafifal.mamun/notebooks/triagerX/slurm_out/%j.out

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate triagerx

python training/developer/developer_training_deeptriage.py --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/deeptriage/developer_sp_tokens.yaml --seed 42