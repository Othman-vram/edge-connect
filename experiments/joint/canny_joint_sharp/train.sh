#!/bin/bash
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 28
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mem=0
#SBATCH --account=m25031

~/edge-connect/.venv/bin/python train.py --model 4 --checkpoints ./experiments/joint/canny_joint_sharp/checkpoints/
