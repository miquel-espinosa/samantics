#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x.%j.out
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --job-name=8gpu-test
#SBATCH --mem=400G
##------------------------ End job description ------------------------

module purge && source /gws/nopw/j04/sensecdt/users/mespi/virtualenvs/ref_sam/bin/activate

export CONFIG_FILE=work_configs/RefSAM_det_coco_base.py
export CHECKPOINT=work_dirs/RefSAM_det_coco_base/epoch_50.pth
export NUM_GPUS=8

sh tools/dist_test.sh $CONFIG_FILE $CHECKPOINT $NUM_GPUS