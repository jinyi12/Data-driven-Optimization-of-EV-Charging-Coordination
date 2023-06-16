#!/bin/env bash
#SBATCH --job-name=pytorch_dev
#SBATCH --partition=gpu.medium
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --time=3-00:00:00
#SBATCH --mem=16GB
#SBATCH --output=/home/yjin0055/jupyter.log
### #SBATCH --nodelist=mum-hpc2-gpu3

module load anaconda/2022.05
module load cuda/
module load cuda/cudnn

eval "$(conda shell.bash hook)"

conda activate optimization

cat /etc/hosts
jupyter lab --ip=0.0.0.0 --port=8888 --notebook-dir="~/Project/DayAheadForecast/" 


