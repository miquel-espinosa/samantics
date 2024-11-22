#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x.%j.out
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --job-name=4gpu-test-large
#SBATCH --mem=400G
##------------------------ End job description ------------------------

module purge && source /gws/nopw/j04/sensecdt/users/mespi/virtualenvs/ref_sam/bin/activate

export CONFIG_FILE=work_configs/RefSAM_det_coco_base_large.py
export CHECKPOINT=work_dirs/RefSAM_det_coco_base_large/epoch_50.pth
export NUM_GPUS=4

sh tools/dist_test.sh $CONFIG_FILE $CHECKPOINT $NUM_GPUS