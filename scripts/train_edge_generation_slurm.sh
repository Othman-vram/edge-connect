#!/bin/bash
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 28
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=0
#SBATCH --account=m25031

~/edge-connect/.venv/bin/python train.py --model 1 --checkpoints ./checkpoints/
