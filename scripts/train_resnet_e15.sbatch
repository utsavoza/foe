#!/bin/bash

#SBATCH --job-name=train-resnet-e15
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=96GB
#SBATCH --priority=4294967293
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --account=class
#SBATCH --out=./logs/train-resnet-e15.out

module purge;

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;

singularity exec --nv \
  --overlay /scratch/$USER/my_env/overlay-50G-10M.ext3:ro \
  /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
  /bin/bash -c "source /ext3/env.sh; python train.py --rotation=360 --v_flip --batch_size=64 --name_append='e10' --min_epochs=1 --max_epochs=15 --early_stopping=5 --model_checkpointing=1"
