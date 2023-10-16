#!/bin/bash
#SBATCH -J JupyterServer
#SBATCH -t 5:00:00
#SBATCH -n 1
#SBATCH --mem=40G
#SBATCH --partition=gpu-a100 --gres=gpu:1
#SBATCH --output=./slurm_out/%j.out

source ~/miniconda3/etc/profile.d/conda.sh && \
conda activate triagerx && \
jupyter notebook --no-browser --ip="*" --port=9779 --NotebookApp.token=letmepass777 --notebook-dir="/home/mdafifal.mamun/notebooks/triagerX/notebook"