#!/bin/sh
#SBATCH --array=1-10
#SBATCH --output=resultGA10-4-26-08.2022-%a.txt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=256

srun python3 solver.py