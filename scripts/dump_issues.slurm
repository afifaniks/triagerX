#!/bin/bash
#SBATCH -J idump
#SBATCH -t 3-00:00:00
#SBATCH -n 1
#SBATCH --mem=5G
#SBATCH --partition=cpu2023
#SBATCH --output=/home/mdafifal.mamun/notebooks/triagerX/slurm_out/dump_%j.out

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate triagerx
python util/issue_dump.py --path /work/disa_lab/afif/projects/openj9/issue_repository --owner eclipse-openj9 --repo openj9