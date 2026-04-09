#!/usr/bin/bash 
#SBATCH -J gambit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ugrad
#SBATCH -t 1-0
#SBATCH -o logs/gambit-%A.out

source $HOME/.cargo/env

cargo build --release


exit 0
