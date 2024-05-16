#!/bin/bash
#SBATCH -J triagerx
#SBATCH -t 0-10:00:00
#SBATCH -n 1
#SBATCH --mem=80G
#SBATCH --partition=gpu-a100 --gres=gpu:1
#SBATCH --output=/home/mdafifal.mamun/notebooks/triagerX/slurm_out/%j.out

source ~/miniconda3/etc/profile.d/conda.sh && \
conda activate triagerx && \
python /home/mdafifal.mamun/notebooks/triagerX/training/developer/developer_training.py --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/developer/developer_sp_tokens.yaml --seed 41

source ~/miniconda3/etc/profile.d/conda.sh && \
conda activate triagerx && \
python /home/mdafifal.mamun/notebooks/triagerX/training/developer/developer_training.py --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/developer/developer_summary_description.yaml --seed 41

source ~/miniconda3/etc/profile.d/conda.sh && \
conda activate triagerx && \
python /home/mdafifal.mamun/notebooks/triagerX/training/developer/developer_training.py --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/developer/developer_summary_sp_tokens.yaml --seed 41

