#!/bin/sh

#SBATCH -J face-attributes-upload     # Job Name
#SBATCH -p batch                    # (debug or batch)
#SBATCH -o upload_log.out           # Output file name
#SBATCH --gres=gpu:1                # Number of GPUs to use
#SBATCH --mail-user ririye@smu.edu
#SBATCH --mail-type=ALL
#SBATCH --mem=240GB

python main.py
