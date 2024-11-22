#!/bin/bash
#SBATCH -J triagerx
#SBATCH -t 23:55:00
#SBATCH -n 1
#SBATCH --mem=50G
#SBATCH --partition=disa --gres=gpu:1
#SBATCH --output=/home/mdafifal.mamun/notebooks/triagerX/slurm_out/%j.out

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate triagerx

# python reproduction/lbtp_distillation.py --dataset_path /home/mdafifal.mamun/notebooks/triagerX/data/openj9/openj9_08122024.csv --model_weights_path /work/disa_lab/projects/triagerx/models/distillation/lbtp_openj9_base.pt
python /home/mdafifal.mamun/notebooks/triagerX/reproduction/lbtp/lbtp_distillation.py --dataset_path /home/mdafifal.mamun/notebooks/triagerX/data/mozilla_firefox/deep_data.csv --model_weights_path /work/disa_lab/projects/triagerx/models/distillation/lbtp_mf_base.pt

python /home/mdafifal.mamun/notebooks/triagerX/reproduction/lbtp/train_lbtp.py --dataset_path /home/mdafifal.mamun/notebooks/triagerX/data/mozilla_firefox/classifier_data_20.csv --embedding_model_weights /work/disa_lab/projects/triagerx/models/distillation/lbtp_mf_base.pt --output_model_weights /work/disa_lab/projects/triagerx/models/distillation/lbtp_mf_block9.pt --block 9 --run_name lbtp_mf_block9 --wandb_project mozilla_firefox