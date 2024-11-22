#!/bin/bash
#SBATCH --job-name=train-sam2-imagenet
#SBATCH --partition=partition
#SBATCH --account=partition
#SBATCH --nodes=2
#SBATCH -w gpuhost009,gpuhost010
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=slurm-sam2_%x_%j.out
#SBATCH --mem=400GB

module purge
source /path/users/user/virtualenvs/eval_image_encoders/bin/activate

export TORCH_CUDNN_SDPA_ENABLED=1

echo "SLURM_JOB_NUM_NODES: ${SLURM_JOB_NUM_NODES}"

srun python train.py \
    --data_dir ./data/imagenet \
    --encoder sam2 \
    --sam_checkpoint ./checkpoints/sam2.1_hiera_tiny.pt \
    --sam2_config ./configs/sam2.1/sam2.1_hiera_t.yaml \
    --batch_size 16 \
    --num_nodes ${SLURM_JOB_NUM_NODES} \
    --devices 4 \
    --strategy ddp