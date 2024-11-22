#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x.%j.out
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --job-name=eval_base
#SBATCH --mem=400G
##------------------------ End job description ------------------------

# module purge && source /gws/nopw/j04/sensecdt/users/mespi/virtualenvs/ref_sam/bin/activate

MODEL_TYPE="base"
MODE="eval_embeds" # Model modes: 'compute_embeds', 'eval_embeds', 'None'
N_SHOT=10

# Orchid settings
# CONFIG_FILE="work_configs/RefSAM_det_coco_base.py"
# CHECKPOINT="work_dirs/RefSAM_det_coco_base/epoch_50.pth"
# VAL_ANN_FILE="/work/scratch-nopw2/mespi/sam-hq-hf/sam-hq-training/data/coco/annotations/instances_val2017.json"
# ROOT_DATA_DIR="/work/scratch-nopw2/mespi/sam-hq-hf/sam-hq-training/data/coco/refsam_eval/\
# /10shot_42seed_selectrandom_sortbyarea_base"

# Eng settings
CONFIG_FILE="work_configs/RefSAM_det_coco_base_eng.py"
CHECKPOINT="/localdisk/data2/Users/s2254242/projects_storage/ref_sam/checkpoints/\
RefSAM_det_coco_base_epoch_50.pth"
VAL_ANN_FILE="/localdisk/data1/Data/COCO/annotations_trainval2017/annotations/instances_val2017.json"
NPY_FILE="/localdisk/data2/Users/s2254242/projects_storage/ref_sam/eval_data/\
10shot_42seed_selectrandom_sortbyarea_base.npy"


# IMPORTANT: Run this on SINGLE GPU when compute_embeds from json
CUDA_VISIBLE_DEVICES=0,1,2,5,6 sh tools/dist_test.sh $CONFIG_FILE $CHECKPOINT 5 \
        --cfg-options \
                model.mode="${MODE}" \
                model.n_shot=${N_SHOT} \
                model.embeds_path="${NPY_FILE}" \
                train_pipeline=None \
                train_dataloader=None \
                train_cfg=None \
                optim_wrapper=None \
                param_scheduler=None \
                test_dataloader.dataset.ann_file="${VAL_ANN_FILE}" \
                test_dataloader.dataset.data_prefix.img="val2017/val2017/" \
                test_dataloader.dataset.with_ref=False \
                test_evaluator.ann_file="${VAL_ANN_FILE}"