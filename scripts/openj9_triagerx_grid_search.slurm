#!/bin/bash
#SBATCH -J trxgridsh
#SBATCH -t 1-00:00:00
#SBATCH -n 1
#SBATCH --mem=50G
#SBATCH --partition=disa --gres=gpu:1
#SBATCH --output=/home/mdafifal.mamun/notebooks/triagerX/slurm_out/openj9_grid_search_%j.out

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate triagerx

python /home/mdafifal.mamun/notebooks/triagerX/triagerx/trainer/grid_search.py