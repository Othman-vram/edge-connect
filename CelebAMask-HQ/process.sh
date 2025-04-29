#!/bin/bash
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 80
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=256
#SBATCH --account=m25031

/home/tompoget/edge-connect/.venv/bin/python /home/tompoget/edge-connect/CelebAMask-HQ/process.py
