#!/bin/bash
#SBATCH -J triagerx
#SBATCH -t 0-6:00:00
#SBATCH -n 1
#SBATCH --mem=80G
#SBATCH --partition=disa --gres=gpu:1
#SBATCH --output=/home/mdafifal.mamun/notebooks/triagerX/slurm_out/%j.out

source ~/miniconda3/etc/profile.d/conda.sh && \
conda activate triagerx && \
python /home/mdafifal.mamun/notebooks/triagerX/training/component/component_training.py --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/component/component_summary_description.yaml --seed 66

source ~/miniconda3/etc/profile.d/conda.sh && \
conda activate triagerx && \
python /home/mdafifal.mamun/notebooks/triagerX/training/component/component_training.py --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/component/component_summary_description.yaml --seed 12

source ~/miniconda3/etc/profile.d/conda.sh && \
conda activate triagerx && \
python /home/mdafifal.mamun/notebooks/triagerX/training/component/component_training.py --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/component/component_summary_description.yaml --seed 92

source ~/miniconda3/etc/profile.d/conda.sh && \
conda activate triagerx && \
python /home/mdafifal.mamun/notebooks/triagerX/training/component/component_training.py --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/component/component_summary_description.yaml --seed 41

source ~/miniconda3/etc/profile.d/conda.sh && \
conda activate triagerx && \
python /home/mdafifal.mamun/notebooks/triagerX/training/component/component_training.py --config /home/mdafifal.mamun/notebooks/triagerX/training/training_config/component/component_summary_description.yaml --seed 42