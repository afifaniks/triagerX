#!/bin/bash
#SBATCH -J triagerx
#SBATCH -t 0-6:00:00
#SBATCH -n 1
#SBATCH --mem=80G
#SBATCH --partition=gpu-a100 --gres=gpu:1
#SBATCH --output=/home/mdafifal.mamun/notebooks/triagerX/slurm_out/%j.out

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate triagerx

# Array of configuration files
configs=(
    "component_raw_data.yaml"
    "component_summary_sp_tokens.yaml"
    "component_sp_tokens.yaml"
    "component_summary_description.yaml"
)

# Array of seeds
seeds=(66)

# Loop through each seed and config combination
for seed in "${seeds[@]}"; do
    for config in "${configs[@]}"; do
        python /home/mdafifal.mamun/notebooks/triagerX/training/component/component_training.py --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/component/$config --seed $seed
    done
done
