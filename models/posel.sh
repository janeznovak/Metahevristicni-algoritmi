#!/bin/bash
#SBATCH --job-name=GWO_GFG
#SBATCH --mem-per-cpu=100MB
#SBATCH --output=GWOGFG.out
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=256

srun python3 solverGWOGFG.py
