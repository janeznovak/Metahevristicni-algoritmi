#!/bin/bash
#SBATCH --job-name=GWO_GFG
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=100MB
#SBATCH --output=GWOGFG.out
#SBATCH --time=00:05:00

srun python3 solverWOAGFG.py
