#!/bin/bash
#SBATCH -J JupyterServer
#SBATCH -t 5:00:00
#SBATCH -n 1
#SBATCH --mem 0
#SBATCH --partition=gpu-a100 --gres=gpu:1

conda activate triagerx
jupyter notebook --no-browser --ip="*" --port=9779 --NotebookApp.token=letmepass777 --notebook-dir="/home/mdafifal.mamun/notebooks/triagerX/notebook"