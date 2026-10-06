#!/bin/bash

#SBATCH -J af3
#SBATCH --account=def-bkuhlman    
#SBATCH --time=00:20:00          
#SBATCH --cpus-per-task=8         
#SBATCH --gpus-per-node=h100:1            
#SBATCH --mem=20G  
#SBATCH --mail-user=tiger_zhang@med.unc.edu
#SBATCH --mail-type=end,fail

start_time=$(date +%s)

# load modules
module load StdEnv/2023 gcc python/3.11.5 scipy-stack/2023b 
module load hmmer rdkit/2024.03.5 cmake
module load cuda/12.2

# activate environment
source /project/def-bkuhlman/group_files/envs/alphafold3_env/bin/activate


# This example makes a prediction with custom protein MSA and templates.
python /project/def-bkuhlman/group_files/alphafold3/run/run_af3.py \
    --json_path ./alphafold_input_custom.json \
    --output_dir ./af3_preds_custom_msa
	
# deactivate environment
deactivate

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "The job took $elapsed_time seconds to complete."