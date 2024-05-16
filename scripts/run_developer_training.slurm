#!/bin/bash
#SBATCH -J triagerx
#SBATCH -t 0-12:00:00
#SBATCH -n 1
#SBATCH --mem=80G
#SBATCH --partition=disa --gres=gpu:1
#SBATCH --output=/home/mdafifal.mamun/notebooks/triagerX/slurm_out/%j.out

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate triagerx

# Array of configuration files
configs=(
    "developer_raw_data.yaml"
    "developer_summary_sp_tokens.yaml"
    "developer_sp_tokens.yaml"
    "developer_summary_description.yaml"
)

# Array of seeds
seeds=(66)

# Loop through each seed and config combination
for seed in "${seeds[@]}"; do
    for config in "${configs[@]}"; do
        python /home/mdafifal.mamun/notebooks/triagerX/training/developer/developer_training.py --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/developer/$config --seed $seed
    done
done
