#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x.%j.out
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --job-name=create_eval_jsons
#SBATCH --mem=400G
##------------------------ End job description ------------------------

module purge && source /gws/nopw/j04/sensecdt/users/mespi/virtualenvs/ref_sam/bin/activate

export ROOT_DIR=/work/scratch-nopw2/mespi/sam-hq-hf/sam-hq-training/data/coco

python create_eval_jsons.py --n_ref 10 --sortby area --seed 42 --save_plots --root $ROOT_DIR

python create_eval_jsons.py --n_ref 10 --sortby area --seed 123 --save_plots --root $ROOT_DIR

python create_eval_jsons.py --n_ref 10 --sortby area --seed 33 --save_plots --root $ROOT_DIR

