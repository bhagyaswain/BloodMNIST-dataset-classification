#!/bin/bash
#SBATCH --job-name=q2
#SBATCH --cpus-per-task=10 # Adjust according to your memory requirements
#SBATCH --mem-per-cpu=20G 
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt

module load Java/17.0.4
module load Anaconda3/2022.05
source activate myspark

# Activate your conda environment if needed
# source activate my_env

spark-submit --driver-memory 20g --executor-memory 20g /users/acp22abj/com6012/acp22abj-COM6012/Q2/Q2_code.py

