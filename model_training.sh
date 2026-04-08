#!/usr/bin/bash

#SBATCH -J Gambit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ugrad_advisor_x_
#SBATCH -w moana_u8
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out


/data/yho7374/anaconda3/bin/conda init
source ~/.bashrc
conda activate training


exit 0
