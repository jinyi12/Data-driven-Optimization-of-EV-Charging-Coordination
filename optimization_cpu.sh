#!/bin/env bash
#SBATCH --job-name=optimization_dev
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=48
#SBATCH --time=3-00:00:00
#SBATCH --mem=16GB
#SBATCH --output=/home/yjin0055/jupyter_cpu.log

module load anaconda/2022.05
module load cuda/
module load cuda/cudnn

eval "$(conda shell.bash hook)"

conda activate optimization

cat /etc/hosts
jupyter lab --ip=0.0.0.0 --port=8888 --notebook-dir="~/Project/DayAheadForecast/" 


