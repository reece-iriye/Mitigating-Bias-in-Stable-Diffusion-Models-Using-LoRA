#!/bin/sh

#SBATCH -J benchmark-diffusion      # Job Name
#SBATCH -p batch                    # (debug or batch)
#SBATCH -o benchmark-diffusion.out # Output file name
#SBATCH --gres=gpu:1                # Number of GPUs to use
#SBATCH --mail-user ririye@smu.edu
#SBATCH --mail-type=ALL
#SBATCH --mem=128GB

pip install -r ../requirements.txt
python main.py
