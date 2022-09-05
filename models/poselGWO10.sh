#!/bin/sh
#SBATCH --array=1-10
#SBATCH --output=resultGWO10-1-300-default-attack-01-09.2022-%a.txt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=256

srun python3 solverGWOGFG.py