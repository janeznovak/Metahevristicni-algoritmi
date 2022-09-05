#!/bin/sh
#SBATCH --array=1-10
#SBATCH --output=resultPSO10-4-300-25-08.2022-%a.txt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=256

srun python3 solverPSONR.py
