#!/bin/bash
#SBATCH --account=def-ycoady
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=08:00:00
module load cuda eigen python/3.6
virtualenv --no-download /home/srose/scratch/vir_env
source /home/srose/scratch/vir_env/bin/activate
python ./train.py --model deeplab --capture historic --label hist_augment_1 --augment True
