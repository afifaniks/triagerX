#!/bin/bash
#SBATCH -J triagerx
#SBATCH -t 1-00:00:00
#SBATCH -n 1
#SBATCH --mem=80G
#SBATCH --partition=disa --gres=gpu:1
#SBATCH --output=/home/mdafifal.mamun/notebooks/triagerX/slurm_out/%j.out

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate triagerx

python /home/mdafifal.mamun/notebooks/triagerX/reproduction/train_lbtp.py \
    --dataset_path /home/mdafifal.mamun/notebooks/triagerX/data/deeptriage/mozilla_core/classifier_data_20.csv \
    --embedding_model_weights /work/disa_lab/projects/triagerx/models/distillation/lbtp_mc_base.pt \
    --block 9 \
    --output_model_weights /work/disa_lab/projects/triagerx/models/lbtp_dt_gc/lbtp_mc_block9.pt