#!/bin/bash
#SBATCH --job-name=train-sam-imagenet
#SBATCH --partition=partition
#SBATCH --account=partition
#SBATCH --nodes=2
#SBATCH -w gpuhost003,gpuhost006
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=slurm-sam_%x_%j.out
#SBATCH --mem=400GB

module purge
source /path/users/user/virtualenvs/eval_image_encoders/bin/activate

export TORCH_CUDNN_SDPA_ENABLED=1

echo "SLURM_JOB_NUM_NODES: ${SLURM_JOB_NUM_NODES}"

srun python train.py \
    --data_dir ./data/imagenet \
    --encoder sam \
    --sam_checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
    --resume_from_checkpoint ./lightning_logs/sam/version_0/checkpoints/last.ckpt \
    --batch_size 16 \
    --num_nodes ${SLURM_JOB_NUM_NODES} \
    --devices 4 \
    --strategy ddp