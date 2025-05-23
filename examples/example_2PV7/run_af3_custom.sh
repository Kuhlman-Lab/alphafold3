#!/bin/bash

#SBATCH -J af3_custom-2PV7
#SBATCH -p a100-gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16g
#SBATCH -t 03-00:00:00
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%j-%x.out
#SBATCH --error=slurm-%j-%x.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=user@email.com

source ~/.bashrc
module load gcc/12.2.0
conda activate af3

# This example makes a prediction for 2PV7 with custom protein MSA and templates.
python /proj/kuhl_lab/alphafold3/run/run_af3.py \
    --json_path ./alphafold_input_custom.json \
    --output_dir ./af3_preds_custom