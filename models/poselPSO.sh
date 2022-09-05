#!/bin/bash
#SBATCH --job-name=PSONR
#SBATCH --mem-per-cpu=1001MB
#SBATCH --output=PSONR.out
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=256
#SBATCH --ntasks=2


srun python3 solverPSONR.py
