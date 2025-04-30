#!/bin/bash
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 100
#SBATCH --gres=gpu:5
#SBATCH --time=20:00:00
#SBATCH --mem=4096
#SBATCH --account=m25031
#SBATCH --exclude=juliet2

~/edge-connect/.venv/bin/python train.py --checkpoints ./experiments/joint/anno_sharp/checkpoints/
