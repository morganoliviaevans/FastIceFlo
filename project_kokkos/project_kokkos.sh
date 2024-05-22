#!/bin/bash

#SBATCH --job-name=project_kokkos
#SBATCH --partition=talon-gpu32
#SBATCH --gres=gpu:1
#SBATCH --time=00:60:00
#SBATCH --ntasks=1
#SBATCH --output=%x.%j.txt

#Optimal PT solver parameter values for the glacier model configuration and the corresponding DoFs. In this case for JKS8e4
damp=0.998;
visc_rela=0.01;
vel_rela=0.995;

#nvcc -O3 -std=c++17 -arch=sm_70  project_kokkos.cpp -o project_kokkos -Ddmp=$damp -Drelax=$vel_rela -Drel=$visc_rela


module load gcc
echo "Job started at $(date)"
./project_kokkos PIG2e6.bin $damp $visc_rela $vel_rela 
echo "Job ended at $(date)"


#Generate NSIGHT Systems report
#nsys profile --stats=true ./project_kokkos

#Generate NSIGHT Compute report
#To avoid generating large-sized (in Bytes) reports, run up to 2 iterations
#ncu --set full --import-source yes -o report ./project_kokkos;
