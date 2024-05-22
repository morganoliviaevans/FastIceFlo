#!/bin/bash

#SBATCH --job-name="project_kokkos"
#SBATCH --output="a.out.%j.%N.txt"
#SBATCH --partition=gpuMI100x8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=4   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bcjy-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH -t 08:00:00

damp=0.998;
visc_rela=0.1;
vel_rela=0.999;
echo project_kokkos AMD_MI100
#srun -N 1 -n 4 ./build/project_kokkos > myjob.out
./build/project_kokkos JKS2e7.bin $damp $visc_rela $vel_rela 
# py-torch example, --ntasks-per-node=1 --cpus-per-task=16
# srun python3 multiple_gpu.py

