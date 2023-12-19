#!/bin/bash

#SBATCH --job-name=craft-attacks-transfer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=96GB
#SBATCH --priority=4294967293
#SBATCH --account=class
#SBATCH --out=./logs/craft-attacks-transfer.out

module purge;

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;

singularity exec --nv \
  --overlay /scratch/$USER/my_env/overlay-50G-10M.ext3:ro \
  /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
  /bin/bash -c "source /ext3/env.sh; python craft_attacks.py --path_model='models/resnet50_weights_final.hdf5' --save_results"
