#!/bin/sh

#SBATCH -J CompDiffusion            # Job Name
#SBATCH -p batch                    # (debug or batch)
#SBATCH -o comp-diffusion.out       # Output file name
#SBATCH --gres=gpu:1                # Number of GPUs to use
#SBATCH --mail-user ririye@smu.edu
#SBATCH --mail-type=ALL
#SBATCH --mem=128GB

python main.py
