#!/bin/bash
#SBATCH -J triagerx
#SBATCH -t 2-00:00:00
#SBATCH -n 1
#SBATCH --mem=92G
#SBATCH --partition=cpu2022
#SBATCH --output=/home/mdafifal.mamun/notebooks/triagerX/slurm_out/tfidf_%j.out

source ~/miniconda3/etc/profile.d/conda.sh && \
conda activate triagerx && \
python /home/mdafifal.mamun/notebooks/triagerX/reproduction/tf_idf_2.py