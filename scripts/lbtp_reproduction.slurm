#!/bin/bash
#SBATCH -J triagerx
#SBATCH -t 1-00:00:00
#SBATCH -n 1
#SBATCH --mem=50G
#SBATCH --partition=gpu-v100 --gres=gpu:1
#SBATCH --output=/home/mdafifal.mamun/notebooks/triagerX/slurm_out/lbtp_%j.out

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate triagerx

# Check the value of the first parameter
if [ "$1" == "mc" ]; then
    python /home/mdafifal.mamun/notebooks/triagerX/reproduction/train_lbtp.py \
        --dataset_path /home/mdafifal.mamun/notebooks/triagerX/data/deeptriage/mozilla_core/classifier_data_20.csv \
        --embedding_model_weights /work/disa_lab/projects/triagerx/models/distillation/lbtp_mc_base.pt \
        --block 9 \
        --output_model_weights /work/disa_lab/projects/triagerx/models/lbtp_dt_gc/lbtp_mc_block9.pt \
        --run_name lbtp_mc \
        --wandb_project mozilla_core

elif [ "$1" == "gc" ]; then
    python /home/mdafifal.mamun/notebooks/triagerX/reproduction/train_lbtp.py \
        --dataset_path /home/mdafifal.mamun/notebooks/triagerX/data/deeptriage/google_chrome/classifier_data_20.csv \
        --embedding_model_weights /work/disa_lab/projects/triagerx/models/distillation/lbtp_gc_base.pt \
        --block 9 \
        --output_model_weights /work/disa_lab/projects/triagerx/models/lbtp_dt_gc/lbtp_gc_block9.pt \
        --run_name lbpt_gc \
        --wandb_project google_chromium

elif [ "$1" == "mf" ]; then
    python /home/mdafifal.mamun/notebooks/triagerX/reproduction/train_lbtp.py \
        --dataset_path /home/mdafifal.mamun/notebooks/triagerX/data/deeptriage/mozilla_firefox/classifier_data_20.csv \
        --embedding_model_weights /work/disa_lab/projects/triagerx/models/distillation/lbtp_mf_base.pt \
        --block 9 \
        --output_model_weights /work/disa_lab/projects/triagerx/models/lbtp_dt_gc/lbtp_mf_block9.pt \
        --run_name lbpt_mf \
        --wandb_project mozilla_firefox

elif [ "$1" == "openj9" ]; then
    python /home/mdafifal.mamun/notebooks/triagerX/reproduction/train_lbtp.py \
        --dataset_path /home/mdafifal.mamun/notebooks/triagerX/data/openj9/openj9_08122024.csv \
        --embedding_model_weights /work/disa_lab/projects/triagerx/models/distillation/lbtp_openj9_base.pt \
        --block 9 \
        --output_model_weights /work/disa_lab/projects/triagerx/models/lbtp_dt_gc/lbtp_openj9_block9.pt \
        --run_name lbpt_openj9 \
        --wandb_project openj9-developers

elif [ "$1" == "ts" ]; then
    python /home/mdafifal.mamun/notebooks/triagerX/reproduction/train_lbtp.py \
        --dataset_path /home/mdafifal.mamun/notebooks/triagerX/data/typescript_last.csv \
        --embedding_model_weights /work/disa_lab/projects/triagerx/models/distillation/lbtp_ts_base.pt \
        --block 9 \
        --output_model_weights /work/disa_lab/projects/triagerx/models/lbtp_dt_gc/lbtp_ts_block9.pt \
        --run_name lbpt_ts \
        --wandb_project microsoft-ts

else
    echo "Invalid parameter. Please use 'mc' or 'gc'."
fi
