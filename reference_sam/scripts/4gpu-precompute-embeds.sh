#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x.%j.out
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --job-name=4gpu-precompute-embeds
#SBATCH --mem=400G
##------------------------ End job description ------------------------

module purge && source /gws/nopw/j04/sensecdt/users/mespi/virtualenvs/ref_sam/bin/activate

export CHECKPOINT="work_dirs/RefSAM_det_coco_base_large/epoch_50.pth"

# First step: Run script to select 10 reference images and create coco-json-like file
export TRAIN_ANN_PATH="/work/scratch-nopw2/mespi/sam-hq-hf/sam-hq-training/data/coco/annotations/instances_train2017.json"
export SAVE_JSON_PATH="/work/scratch-nopw2/mespi/sam-hq-hf/sam-hq-training/data/coco/precompute_10_shot_annotations_ref_train.json"
# Run once, otherwise comment out
# python create_ref_json.py $TRAIN_ANN_PATH $SAVE_JSON_PATH

# Second step: Precompute the embeddings for the selected reference images and store in npy file
# IMPORTANT: Run this on single GPU
export PRECOMPUTE_CONFIG_FILE="work_configs/RefSAM_det_coco_large_precompute.py"
# Run once, otherwise comment out
sh tools/dist_test.sh $PRECOMPUTE_CONFIG_FILE $CHECKPOINT 1 # Single GPU