#!/bin/sh
#SBATCH --partition=slurm_shortgpu
#SBATCH --time=0-00:05:00 # run time in days-hh:mm:ss
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8  # Use 8 OpenMP threads (you can use however many you like)
#SBATCH --error=/srv/home/thaines/HW12/problem1.err
#SBATCH --output=/srv/home/thaines/HW12/problem1.out
##SBATCH --gpu:res=1

mpiexec -np 2 ./problem1 1024
