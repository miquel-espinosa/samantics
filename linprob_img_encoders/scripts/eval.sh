#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=partition
#SBATCH --account=partition
#SBATCH --nodes=1
#SBATCH -w gpuhost003
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=slurm-eval_%x_%j.out
#SBATCH --mem=400GB

module purge
source /path/users/user/virtualenvs/eval_image_encoders/bin/activate

export TORCH_CUDNN_SDPA_ENABLED=1

echo "SLURM_JOB_NUM_NODES: ${SLURM_JOB_NUM_NODES}"

srun python train.py \
    --eval_only \
    --data_dir ./data/imagenet \
    --encoder clip \
    --checkpoint_path ./lightning_logs/clip/version_0/checkpoints/last.ckpt \
    --batch_size 256