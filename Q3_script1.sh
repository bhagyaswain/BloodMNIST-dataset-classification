#!/bin/bash
#SBATCH --job-name=q3_grid
#SBATCH --cpus-per-task=10 # Adjust according to your memory requirements
#SBATCH --mem-per-cpu=20G
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
# Load Java module
module load Java/17.0.4

# Load Anaconda module
module load Anaconda3/2022.05

# Activate your Anaconda environment
source activate myspark
# Run your Spark script
spark-submit --driver-memory 20g --executor-memory 20g /users/acp22abj/com6012/acp22abj-COM6012/Q3/Q3_code1.py
