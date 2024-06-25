#!/bin/bash
#SBATCH -J deeptriagerx
#SBATCH -t 1-00:00:00
#SBATCH -n 1
#SBATCH --mem=50G
#SBATCH --partition=gpu-v100 --gres=gpu:1
#SBATCH --output=/home/mdafifal.mamun/notebooks/triagerX/slurm_out/dt_%j.out

# Check if the first parameter is "gc" or "mc"
CONFIG_PARAM=$1

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate triagerx

# Run the appropriate configuration based on the parameter
if [ "$CONFIG_PARAM" == "gc-cnn" ]; then
    echo "Running GC (CNN Transformer) configuration"
    python /home/mdafifal.mamun/notebooks/triagerX/training/developer/developer_training_deeptriage.py \
        --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/deeptriage/cnn_transformer.yaml \
        --dataset_path /home/mdafifal.mamun/notebooks/triagerX/data/deeptriage/google_chrome/classifier_data_20.csv \
        --seed 42

elif [ "$CONFIG_PARAM" == "gc-triagerx" ]; then
    echo "Running GC (Triagerx Model) configuration"
    python /home/mdafifal.mamun/notebooks/triagerX/training/developer/developer_training_deeptriage.py \
        --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/deeptriage/triagerx.yaml \
        --dataset_path /home/mdafifal.mamun/notebooks/triagerX/data/deeptriage/google_chrome/classifier_data_20.csv \
        --seed 42
        
elif [ "$CONFIG_PARAM" == "mc-triagerx" ]; then
    echo "Running Mozilla Core (Triagerx Model) configuration"
    python /home/mdafifal.mamun/notebooks/triagerX/training/developer/developer_training_deeptriage.py \
        --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/deeptriage/triagerx.yaml \
        --dataset_path /home/mdafifal.mamun/notebooks/triagerX/data/deeptriage/mozilla_core/classifier_data_20.csv \
        --seed 42

elif [ "$CONFIG_PARAM" == "mc-cnn" ]; then
    echo "Running Mozilla Core (CNN Transformer) configuration"
    python /home/mdafifal.mamun/notebooks/triagerX/training/developer/developer_training_deeptriage.py \
        --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/deeptriage/cnn_transformer.yaml \
        --dataset_path /home/mdafifal.mamun/notebooks/triagerX/data/deeptriage/mozilla_core/classifier_data_20.csv \
        --seed 42

elif [ "$CONFIG_PARAM" == "mf-triagerx" ]; then
    echo "Running MF (Triagerx Model) configuration"
    python /home/mdafifal.mamun/notebooks/triagerX/training/developer/developer_training_deeptriage.py \
        --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/deeptriage/triagerx.yaml \
        --dataset_path /home/mdafifal.mamun/notebooks/triagerX/data/deeptriage/mozilla_firefox/classifier_data_20.csv \
        --seed 42

elif [ "$CONFIG_PARAM" == "mf-cnn" ]; then
    echo "Running MF (Triagerx Model) configuration"
    python /home/mdafifal.mamun/notebooks/triagerX/training/developer/developer_training_deeptriage.py \
        --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/deeptriage/cnn_transformer.yaml \
        --dataset_path /home/mdafifal.mamun/notebooks/triagerX/data/deeptriage/mozilla_firefox/classifier_data_20.csv \
        --seed 42

else
    echo "Invalid parameter. Use 'gc-{X}' for GC configuration, 'mf-{X}' for Mozilla Firefox configuration or 'mc-{X}' for Mozilla Core configuration."
fi
