#!/bin/bash
#SBATCH -J triagerx
#SBATCH -t 2-12:00:00
#SBATCH -n 1
#SBATCH --mem=80G
#SBATCH --partition=disa --gres=gpu:1
#SBATCH --output=/home/mdafifal.mamun/notebooks/triagerX/slurm_out/component_%j.out

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate triagerx

# python /home/mdafifal.mamun/notebooks/triagerX/notebook/openj9/data_aug.py
python /home/mdafifal.mamun/notebooks/triagerX/training/component/component_training.py --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/openj9/component/cnn_transformer.yaml --dataset_path /home/mdafifal.mamun/notebooks/triagerX/data/openj9/openj9_22112024.csv --seed 42