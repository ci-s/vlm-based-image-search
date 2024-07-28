#!/bin/bash
#SBATCH --job-name="llava"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=48G
#SBATCH --time=0-08:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/usr/prakt/s0070/vlm-based-image-search/logs/slurm-%j.out
#SBATCH --part=PRACT --qos=practical_course

export NCCL_DEBUG=INFO
export HYDRA_FULL_ERROR=1
pwd; hostname; date
nvidia-smi

python create_captions.py
