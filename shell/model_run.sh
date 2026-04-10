#!/usr/bin/bash 
#SBATCH -J gambit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad_advisor_x 
#SBATCH -w moana-u8
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out


/data/yho7374/anaconda3/bin/conda init
source ~/.bashrc
conda activate training

cd /data/yho7374/repos/chess_rust/shell
./preprocess.sh
./model_training.sh
./rust_test.sh