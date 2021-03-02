#!/usr/bin/env bash

#SBATCH --output logs/trashcan-%J.log  
#SBATCH --job-name trashcan		# good manners rule
#SBATCH --partition gpu	# or gpu_small
#SBATCH --gpus 4	# or gpu_small
####SBATCH --nodes	2 # amount of nodes allocated (same as –N)
#######SBATCH --cpus-per-task 2
######SBATCH --ntasks 4   # number of tasks to launch (overall, same as -n)
#####SBATCH --gres	gpu:4	# number of GPUs to use (Per node!) Max 4 per node
#####SBATCH --time	01-00:00:00	# walltime (less requested time -> less time in queue)

set -x
module rm *
module add python/anaconda3
module add gpu/cuda-10.0
module add compilers/gcc-5.5.0
module load mpi/openmpi-3.1.2
CONFIG=$1
CHECKPOINT=$2

srun python -u tools/train.py ${CONFIG} --resume=${CHECKPOINT} --launcher="slurm" 
