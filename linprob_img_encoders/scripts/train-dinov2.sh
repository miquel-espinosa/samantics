#!/bin/bash
#SBATCH --job-name=train-dinov2-imagenet
#SBATCH --partition=partition
#SBATCH --account=partition
#SBATCH --nodes=1
#SBATCH -w gpuhost005
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=slurm-dinov2_%x_%j.out
#SBATCH --mem=400GB

module purge
source /path/users/user/virtualenvs/eval_image_encoders/bin/activate

export TORCH_CUDNN_SDPA_ENABLED=1

echo "SLURM_JOB_NUM_NODES: ${SLURM_JOB_NUM_NODES}"

srun python train.py \
    --data_dir ./data/imagenet \
    --encoder dinov2 \
    --batch_size 32 \
    --num_nodes ${SLURM_JOB_NUM_NODES} \
    --devices 4 \
    --strategy ddp