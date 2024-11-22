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

# module purge && source /gws/nopw/j04/sensecdt/users/mespi/virtualenvs/ref_sam/bin/activate

# Orchid settings
# CHECKPOINT="work_dirs/RefSAM_det_coco_base/epoch_50.pth"
# CONFIG_FILE="work_configs/RefSAM_det_coco_base.py"
# ROOT_DATA_DIR="/work/scratch-nopw2/mespi/sam-hq-hf/sam-hq-training/data/coco/refsam_eval"

# Eng settings
CONFIG_FILE="work_configs/RefSAM_det_coco_base_eng.py"
CHECKPOINT="/localdisk/data2/Users/s2254242/projects_storage/ref_sam/checkpoints/RefSAM_det_coco_base_epoch_50.pth"
ROOT_DATA_DIR="/localdisk/data2/Users/s2254242/projects_storage/ref_sam/eval_data"

NAME="10shot_42seed_selectrandom_sortbyarea"

# Model modes: 'compute_embeds', 'eval_embeds'

# IMPORTANT: Run this on SINGLE GPU when compute_embeds from json
CUDA_VISIBLE_DEVICES=6 sh tools/dist_test.sh $CONFIG_FILE $CHECKPOINT 1 \
        --cfg-options model.mode="compute_embeds" \
                model.n_shot=10 \
                model.embeds_path="/localdisk/data2/Users/s2254242/projects_storage/ref_sam/eval_data/10shot_42seed_selectrandom_sortbyarea_base.npy" \
                train_pipeline=None \
                train_dataloader=None \
                train_cfg=None \
                optim_wrapper=None \
                param_scheduler=None \
                val_dataloader.dataset.ref_equal_target=True \
                val_dataloader.dataset.ann_file="/localdisk/data2/Users/s2254242/projects_storage/ref_sam/eval_data/10shot_42seed_selectrandom_sortbyarea.json" \
                val_dataloader.dataset.data_prefix.img="train2017/train2017/" \
                val_evaluator.ann_file="/localdisk/data2/Users/s2254242/projects_storage/ref_sam/eval_data/10shot_42seed_selectrandom_sortbyarea.json"
