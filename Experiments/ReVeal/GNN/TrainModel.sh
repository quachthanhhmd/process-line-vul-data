#!/bin/bash
#SBATCH --account=def-m2nagapp
#SBATCH --ntasks=1               # number of MPI processes
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G      # memory; default unit is megabytes
#SBATCH --gres=gpu:v100l:1
#SBATCH --time=48:20:00           # time (DD-HH:MM)
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
export LD_LIBRARY_PATH=$EBROOTCUDA/lib64:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=1; python /home/partha9/Devign/main.py
