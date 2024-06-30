#!/bin/bash
#SBATCH -J deeptriagerx
#SBATCH -t 1-00:00:00
#SBATCH -n 1
#SBATCH --mem=50G
#SBATCH --partition=gpu-v100 --gres=gpu:1
#SBATCH --output=/home/mdafifal.mamun/notebooks/triagerX/slurm_out/ms_%j.out

CONFIG_PARAM=$1

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate triagerx

# Run the appropriate configuration based on the parameter
if [ "$CONFIG_PARAM" == "pt-triagerx" ]; then
    echo "Running PowerToys (Triagerx Model) configuration"
    python /home/mdafifal.mamun/notebooks/triagerX/training/developer/developer_training_powertoys.py \
        --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/microsoft/triagerx.yaml \
        --dataset_path /home/mdafifal.mamun/notebooks/triagerX/data/powertoys_last.csv \
        --seed 42

elif [ "$CONFIG_PARAM" == "ts-triagerx" ]; then
    echo "Running Typescript (Triagerx Model) configuration"
    python /home/mdafifal.mamun/notebooks/triagerX/training/developer/developer_training_typescript.py \
        --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/microsoft/ts-triagerx.yaml \
        --dataset_path /home/mdafifal.mamun/notebooks/triagerX/data/typescript_last.csv \
        --seed 42

elif [ "$CONFIG_PARAM" == "ts-cnn" ]; then
    echo "Running Typescript (Triagerx Model) configuration"
    python /home/mdafifal.mamun/notebooks/triagerX/training/developer/developer_training_typescript.py \
        --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/microsoft/ts-cnn_transformer.yaml \
        --dataset_path /home/mdafifal.mamun/notebooks/triagerX/data/typescript_last.csv \
        --seed 42

else
    echo "Invalid parameter."
fi
