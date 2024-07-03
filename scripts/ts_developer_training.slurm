#!/bin/bash
#SBATCH -J deeptriagerx
#SBATCH -t 1-00:00:00
#SBATCH -n 1
#SBATCH --mem=50G
#SBATCH --partition=disa --gres=gpu:1
#SBATCH --output=/home/mdafifal.mamun/notebooks/triagerX/slurm_out/ms_%j.out

CONFIG_PARAM=$1

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate triagerx

if [ "$CONFIG_PARAM" == "triagerx" ]; then
    echo "Running Typescript (Triagerx Model) configuration"
    python /home/mdafifal.mamun/notebooks/triagerX/training/developer/developer_training_typescript.py \
        --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/microsoft/ts-triagerx.yaml \
        --dataset_path /home/mdafifal.mamun/notebooks/triagerX/data/typescript_last.csv \
        --seed 42

elif [ "$CONFIG_PARAM" == "cnn" ]; then
    echo "Running Typescript (Triagerx Model) configuration"
    python /home/mdafifal.mamun/notebooks/triagerX/training/developer/developer_training_typescript.py \
        --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/microsoft/ts-cnn_transformer.yaml \
        --dataset_path /home/mdafifal.mamun/notebooks/triagerX/data/typescript_last.csv \
        --seed 42

elif [ "$CONFIG_PARAM" == "fcn" ]; then
    echo "Running Typescript (Triagerx Model) configuration"
    python /home/mdafifal.mamun/notebooks/triagerX/training/developer/developer_training_typescript.py \
        --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/microsoft/ts-fcn_transformer.yaml \
        --dataset_path /home/mdafifal.mamun/notebooks/triagerX/data/typescript_last.csv \
        --seed 42

else
    echo "Invalid parameter."
fi
