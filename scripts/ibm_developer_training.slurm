#!/bin/bash
#SBATCH -J deeptriagerx
#SBATCH -t 1-00:00:00
#SBATCH -n 1
#SBATCH --mem=50G
#SBATCH --partition=disa --gres=gpu:1
#SBATCH --output=/home/mdafifal.mamun/notebooks/triagerX/slurm_out/ibm_%j.out

CONFIG_PARAM=$1

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate triagerx

# Run the appropriate configuration based on the parameter
if [ "$CONFIG_PARAM" == "triagerx" ]; then
    echo "Running Openj9 (Triagerx Model) configuration"
    python /home/mdafifal.mamun/notebooks/triagerX/training/developer/developer_training_ibm_openj9.py \
        --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/ibm/developer_triagerx.yaml \
        --dataset_path /home/mdafifal.mamun/notebooks/triagerX/data/openj9/openj9_08122024.csv \
        --seed 42

elif [ "$CONFIG_PARAM" == "component" ]; then
    echo "Running Openj9(CNN Transformer Model) Component configuration"
    python /home/mdafifal.mamun/notebooks/triagerX/training/component/component_training.py --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/openj9/component/cnn_transformer.yaml --seed 42

elif [ "$CONFIG_PARAM" == "fcn" ]; then
    echo "Running Openj9(FCN Transformer Model) configuration"
    python /home/mdafifal.mamun/notebooks/triagerX/training/developer/developer_training_openj9.py \
        --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/openj9/developer/fcn_transformer.yaml \
        --dataset_path /home/mdafifal.mamun/notebooks/triagerX/data/openj9_dataset_6_7_24_last.csv \
        --seed 42

else
    echo "Invalid parameter. Use 'triagerx' for Openj9 (Triagerx) configuration or 'cnn' for Openj9 (CNN-Transformer) configuration."
fi


python /home/mdafifal.mamun/notebooks/triagerX/training/developer/developer_training_torch.py \
--config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/torch/triagerx.yaml \
--dataset_path /home/mdafifal.mamun/notebooks/triagerX/data/pytorch_last.csv \
--seed 42