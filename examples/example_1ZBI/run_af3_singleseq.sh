#!/bin/bash

#SBATCH -J af3_singleseq-1ZBI
#SBATCH -p main
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16g
#SBATCH -t 03-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%j-%x.out
#SBATCH --error=slurm-%j-%x.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=user@email.com

source ~/.bashrc
#module load gcc/12.2.0
conda activate af3

# Work around for a known XLA issue:
# https://github.com/google-deepmind/alphafold3/blob/main/docs/performance.md#compilation-time-workaround-with-xla-flags
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"

export PATH="/spshared/apps/miniconda3/envs/af3/bin/:$PATH"
export LD_PATH="/spshared/apps/miniconda3/envs/af3/lib:$LD_PATH"
export CUDA_HOME="/spshared/apps/miniconda3/envs/af3"

# This example makes a homodimer prediction with no MSA or templates.
python /spshared/apps/alphafold3/run/run_af3.py \
    --json_path ./alphafold_input.json \
    --output_dir ./af3_preds_singleseq \
    --flash_attention_implementation xla