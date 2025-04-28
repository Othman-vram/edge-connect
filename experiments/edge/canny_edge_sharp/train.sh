#!/bin/bash
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 60
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mem=256
#SBATCH --account=m25031

~/edge-connect/.venv/bin/python train.py --checkpoints ./experiments/edge/canny_edge_sharp/checkpoints/
