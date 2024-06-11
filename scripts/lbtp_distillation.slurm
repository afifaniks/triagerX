#!/bin/bash
#SBATCH -J triagerx
#SBATCH -t 0-14:00:00
#SBATCH -n 1
#SBATCH --mem=50G
#SBATCH --partition=disa --gres=gpu:1
#SBATCH --output=/home/mdafifal.mamun/notebooks/triagerX/slurm_out/%j.out

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate triagerx

python reproduction/lbtp_distillation.py --dataset_path /home/mdafifal.mamun/notebooks/triagerX/data/deeptriage/mozilla_core/deep_data.csv --model_weights_path /work/disa_lab/projects/triagerx/models/distillation/lbtp_mc_base.pt