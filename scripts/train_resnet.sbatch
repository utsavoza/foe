#!/bin/bash

#SBATCH --job-name=train-resnet-v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=150:00:00
#SBATCH --mem=96GB
#SBATCH --priority=4294967293
#SBATCH --gres=gpu:1
#SBATCH --account=class
#SBATCH --out=./logs/train-resnet-v2.out

module purge;

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;

singularity exec --nv \
  --overlay /scratch/$USER/my_env/overlay-50G-10M.ext3:ro \
  /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
  /bin/bash -c "source /ext3/env.sh; python train.py --rotation=360 --v_flip --batch_size=128 --name_append='v2' --min_epochs=1 --early_stopping=10 --model_checkpointing=5"
