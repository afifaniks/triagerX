#!/bin/bash
#SBATCH -J triagerx
#SBATCH -t 1-00:00:00
#SBATCH -n 1
#SBATCH --mem=5G
#SBATCH --partition=cpu2023
#SBATCH --output=/home/mdafifal.mamun/notebooks/triagerX/slurm_out/dump_%j.out

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate triagerx
python /home/mdafifal.mamun/notebooks/triagerX/util/issue_dump.py --path /home/mdafifal.mamun/notebooks/triagerX/data/openj9_issue_data