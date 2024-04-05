#!/bin/bash
#SBATCH --job-name="ssa_fem_pt"
#SBATCH --output="ssa_fem_pt.%j.%N.out"
#SBATCH --partition=gpuA100x4
#SBATCH --mem=50G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1       # could be 1 for py-torch
#SBATCH --cpus-per-task=1         # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest        # select a cpu close to gpu on pci bus topology
#SBATCH --account=bcjy-delta-gpu  # <- match to a "Project" returned by the "accounts" command
#SBATCH --exclusive               # dedicated node for this job
#SBATCH --no-requeue
#SBATCH -t 00:10:00

export OMP_NUM_THREADS=1  # if code is not multithreaded, otherwise set to 8 or 16

# Compile ssa_fem_pt.cpp
ml nvhpc/24.1
nvc++ -O3 -acc -use_fast_math ssa_fem_pt.cpp -o ssa_fem_pt.out

# Run ssa_fem_pt.cpp
nvidia-smi
# srun -N 1 -n 4 ./ssa_fem_pt.out

# Generate Nsight Systems report
nsys profile --stats=true ./ssa_fem_pt.out