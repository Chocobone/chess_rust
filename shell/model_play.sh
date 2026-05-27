#SBATCH -J Euwe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ugrad_advisor_x
#SBATCH -w moana-u1
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out


source /data/yho7374/anaconda3/etc/profile.d/conda.sh
conda activate training

python /data/yho7374/repos/chess_rust/3.playing_test/play.py /data/yho7374/repos/chess_rust/2.training/data/model_final.pt

python /data/yho7374/repos/chess_rust/3.playing_test/play.py /data/yho7374/repos/chess_rust/2.training/data/model_final.pt --color black
