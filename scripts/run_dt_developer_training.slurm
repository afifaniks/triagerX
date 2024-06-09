#!/bin/bash
#SBATCH -J deeptriagerx
#SBATCH -t 1-00:00:00
#SBATCH -n 1
#SBATCH --mem=50G
#SBATCH --partition=disa --gres=gpu:1
#SBATCH --output=/home/mdafifal.mamun/notebooks/triagerX/slurm_out/%j.out

# Check if the first parameter is "gc" or "mc"
CONFIG_PARAM=$1

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate triagerx

# Run the appropriate configuration based on the parameter
if [ "$CONFIG_PARAM" == "gc" ]; then
    echo "Running GC configuration"
    python training/developer/developer_training_deeptriage.py --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/deeptriage/train_gc.yaml --seed 42
elif [ "$CONFIG_PARAM" == "mc-triagerx" ]; then
    echo "Running Mozilla Core (Triagerx Model) configuration"
    python /home/mdafifal.mamun/notebooks/triagerX/training/developer/developer_training_deeptriage.py --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/deeptriage/mc_trx.yaml --seed 42

elif [ "$CONFIG_PARAM" == "mc-cnn" ]; then
    echo "Running Mozilla Core (CNN Transformer Model) configuration"
    python /home/mdafifal.mamun/notebooks/triagerX/training/developer/developer_training_deeptriage.py --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/deeptriage/mc_cnn_transformer.yaml --seed 42
else
    echo "Invalid parameter. Use 'gc' for GC configuration or 'mc' for Mozilla Core configuration."
fi
