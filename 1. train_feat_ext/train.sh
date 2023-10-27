#!/bin/bash
#SBATCH -p part1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 5
#SBATCH -o stdout_%j.txt
#SBATCH -e stderr_%j.txt
#SBATCH --gres=gpu:1

source activate skj_torch1.11_py3.8
python train.py
conda deactivate
