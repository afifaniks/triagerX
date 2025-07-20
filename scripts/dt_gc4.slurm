#!/bin/bash
#SBATCH -J deeptriagerx
#SBATCH -t 1-00:00:00
#SBATCH -n 1
#SBATCH --mem=64G
#SBATCH --partition=disa,gpu-v100,gpu-a100 --gres=gpu:1
#SBATCH --output=/home/mdafifal.mamun/notebooks/triagerX/slurm_out/dt_gc_25_%j.out

# Check if the first parameter is "gc" or "mc"
CONFIG_PARAM=$1

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate triagerx

# python /home/mdafifal.mamun/notebooks/triagerX/triagerx/trainer/grid_search.py
# python /home/mdafifal.mamun/notebooks/triagerX/training/developer/statistical_test_openj9.py --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/openj9/developer/triagerx_stat_test.yaml --dataset_path /home/mdafifal.mamun/notebooks/triagerX/data/openj9/openj9_08122024.csv --seed 42

# echo "Running GC Codebert + RoBERTa"
# python /home/mdafifal.mamun/notebooks/triagerX/training/developer/developer_training_deeptriage.py \
# --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/deeptriage/gc-triagerx.yaml \
# --dataset_path /home/mdafifal.mamun/notebooks/triagerX/data/google_chromium/classifier_data_20.csv \
# --seed 42

# echo "Running GC Codebert + DeBERTa"
# python /home/mdafifal.mamun/notebooks/triagerX/training/developer/developer_training_deeptriage.py \
# --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/deeptriage/gc-triagerx_2.yaml \
# --dataset_path /home/mdafifal.mamun/notebooks/triagerX/data/google_chromium/classifier_data_20.csv \
# --seed 42

# echo "Running Openj9 Codebert + Deberta"
# python /home/mdafifal.mamun/notebooks/triagerX/training/developer/developer_training_openj9.py \
# --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/openj9/developer/triagerx.yaml \
# --dataset_path /home/mdafifal.mamun/notebooks/triagerX/data/openj9/openj9_08122024.csv \
# --seed 42

# echo "Running Openj9 Codebert + Roberta"
# python /home/mdafifal.mamun/notebooks/triagerX/training/developer/developer_training_openj9.py \
# --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/openj9/developer/triagerx_2.yaml \
# --dataset_path /home/mdafifal.mamun/notebooks/triagerX/data/openj9/openj9_08122024.csv \
# --seed 42



# python /home/mdafifal.mamun/notebooks/triagerX/training/developer/statistical_test_openj9.py \
# --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/openj9/developer/triagerx.yaml \
# --dataset_path /home/mdafifal.mamun/notebooks/triagerX/data/openj9/openj9_08122024.csv \
# --seed 42

# python /home/mdafifal.mamun/notebooks/triagerX/training/developer/developer_training_deeptriage.py --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/deeptriage/gc-triagerx.yaml --dataset_path /home/mdafifal.mamun/notebooks/triagerX/data/google_chromium/classifier_data_0.csv --seed 42 --threshold 0

# python /home/mdafifal.mamun/notebooks/triagerX/training/developer/developer_training_deeptriage.py --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/deeptriage/gc-triagerx.yaml --dataset_path /home/mdafifal.mamun/notebooks/triagerX/data/google_chromium/classifier_data_0.csv --seed 42 --threshold 5

# python /home/mdafifal.mamun/notebooks/triagerX/training/developer/developer_training_deeptriage.py --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/deeptriage/gc-triagerx.yaml --dataset_path /home/mdafifal.mamun/notebooks/triagerX/data/google_chromium/classifier_data_0.csv --seed 42 --threshold 10

python /home/mdafifal.mamun/notebooks/triagerX/training/developer/developer_training_deeptriage.py --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/deeptriage/gc-triagerx.yaml --dataset_path /home/mdafifal.mamun/notebooks/triagerX/data/google_chromium/classifier_data_0.csv --seed 42 --threshold 25

# python /home/mdafifal.mamun/notebooks/triagerX/training/developer/statistical_test_openj9.py --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/openj9/developer/triagerx.yaml --dataset_path /home/mdafifal.mamun/notebooks/triagerX/data/openj9/openj9_08122024.csv --seed 42