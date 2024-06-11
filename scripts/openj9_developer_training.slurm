#!/bin/bash
#SBATCH -J deeptriagerx
#SBATCH -t 1-00:00:00
#SBATCH -n 1
#SBATCH --mem=50G
#SBATCH --partition=gpu-v100 --gres=gpu:1
#SBATCH --output=/home/mdafifal.mamun/notebooks/triagerX/slurm_out/openj9_%j.out

CONFIG_PARAM=$1

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate triagerx

# Run the appropriate configuration based on the parameter
if [ "$CONFIG_PARAM" == "triagerx" ]; then
    echo "Running Openj9 (Triagerx Model) configuration"
    python /home/mdafifal.mamun/notebooks/triagerX/training/developer/developer_training_openj9.py \
        --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/openj9/developer/triagerx.yaml \
        --dataset_path /home/mdafifal.mamun/notebooks/triagerX/data/openj9_dataset_6_7_24.csv \
        --seed 42

elif [ "$CONFIG_PARAM" == "cnn" ]; then
    echo "Running Openj9(CNN Transformer Model) configuration"
    python /home/mdafifal.mamun/notebooks/triagerX/training/developer/developer_training_openj9.py \
        --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/openj9/developer/cnn_transformer.yaml \
        --dataset_path /home/mdafifal.mamun/notebooks/triagerX/data/openj9_dataset_6_7_24.csv \
        --seed 42
else
    echo "Invalid parameter. Use 'triagerx' for Openj9 (Triagerx) configuration or 'cnn' for Openj9 (CNN-Transformer) configuration."
fi
