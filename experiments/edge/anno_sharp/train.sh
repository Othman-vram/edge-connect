#!/bin/bash
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 60
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mem=512
#SBATCH --account=m25031
#SBATCH --exclude=juliet2

~/edge-connect/.venv/bin/python train.py --checkpoints ./experiments/edge/anno_sharp/checkpoints/
