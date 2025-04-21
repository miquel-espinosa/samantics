<div align="center">

# There is no SAMantics!
## Exploring SAM as a Backbone for Visual Understanding Tasks

[Arxiv Paper](https://arxiv.org/abs/2411.15288) | [Project Page](https://miquel-espinosa.github.io/samantics/)

<!-- https://github.com/user-attachments/assets/df92a34c-cefb-4f24-aaa6-b1bf38adbbe1 -->


</div>

> _The Segment Anything Model (SAM) was originally designed for label-agnostic mask generation. Does this model also possess inherent semantic understanding, of value to broader visual tasks? In this work we follow a multi-staged approach towards exploring this question. We firstly quantify SAM's semantic capabilities by comparing base image encoder efficacy under classification tasks, in comparison with established models (CLIP and DINOv2). Our findings reveal a significant lack of semantic discriminability in SAM feature representations, limiting potential for tasks that require class differentiation. This initial result motivates our exploratory study that attempts to enable semantic information via in-context learning with lightweight fine-tuning where we observe that generalisability to unseen classes remains limited. Our observations culminate in the proposal of a training-free approach that leverages DINOv2 features, towards better endowing SAM with semantic understanding and achieving instance-level class differentiation through feature-based similarity. Our study suggests that incorporation of external semantic sources provides a promising direction for the enhancement of SAM's utility with respect to complex visual tasks that require semantic understanding._



## Quantifying Semantics in SAM
Benchmarking SAM feature representations against popular vision encoders (CLIP and DINOv2) on ImageNet1K classification, measuring the presence of semantics via linear probing on respective features.

### Installation instructions
```shell
pip install torch torhcvision pytorch_lightning timm torchmetrics tensorboard
```

### Training ImageNet1K linear probing
Code for training and evaluating linear classifiers on frozen vision encoders can be found in `linprob_img_encoders/*`.


## Recovering Semantics in SAM
In-context learning with reference images for SAM fine-tuning to capture class-specific information. 

### Installation instructions

```shell
conda create -n ref_sam python=3.8 -y
source activate ref_sam

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html --no-cache

pip install openmim
mim install mmengine
mim install "mmcv>=2.0.0"

pip install lvis

pip install -e .
```

### Fine-tuning SAM for in-context semantic prompting

Code for fine-tuning SAM for in-context semantic prompting can be found in `reference_sam/*`.

#### Training

```shell
python tools/train.py $CONFIG_FILE
```
For multi-node training, see scripts in `scripts/train/*`

#### Testing
Three steps are required to evaluate the fine-tuned SAM model:
1. Create a new json selecting the reference images for the COCO dataset.
```shell
python create_eval_jsons.py --n_ref 10 --sortby area --seed 42 --save_plots --root $ROOT_DIR
```
2. Pre-compute embeddings for the reference images
```shell
CONFIG_FILE="path_to_config_file"
MODEL_CHECKPOINT="path_to_checkpoint"
EMBEDS_SAVE_PATH="path_to_save_embeds"
JSON_PATH="path_to_json"
IMAGES_PREFIX="prefix_to_images"
NUM_CATEGORIES="num_categories_dataset"
N_SHOT=10
sh tools/dist_test.sh $CONFIG_FILE $MODEL_CHECKPOINT 1 \
    --cfg-options model.mode="compute_embeds" \
        model.n_shot=${N_SHOT} \
        model.embeds_path="${EMBEDS_SAVE_PATH}" \
        ${NUM_CATEGORIES} \
        train_pipeline=None \
        train_dataloader=None \
        train_cfg=None \
        optim_wrapper=None \
        param_scheduler=None \
        test_dataloader.dataset.ref_equal_target=True \
        test_dataloader.dataset.ann_file="${JSON_PATH}" \
        test_dataloader.dataset.data_prefix.img="${IMAGES_PREFIX}/" \
        test_evaluator.ann_file="${JSON_PATH}"
```
3. Evaluate on dataset the precomputed embeddings
```shell
CONFIG_FILE="path_to_config_file"
CHECKPOINT="path_to_checkpoint"
NPY_FILE="path_to_npy_file"
VAL_ANN_FILE="path_to_val_ann_file"
IMAGES_PREFIX="prefix_to_images"
N_SHOT=10
VISUALISATION=False
GTBBOXES=False
CUSTOM_NUM_CATEGORIES=10
CUSTOM_CLASSWISE=False
sh tools/dist_test.sh $CONFIG_FILE $CHECKPOINT 4 \
    --cfg-options \
        model.mode="eval_embeds" \
        model.n_shot=${N_SHOT} \
        model.embeds_path="${NPY_FILE}" \
        ${CUSTOM_NUM_CATEGORIES} \
        ${GTBBOXES} \
        ${VISUALISATION} \
        train_pipeline=None \
        train_dataloader=None \
        train_cfg=None \
        optim_wrapper=None \
        param_scheduler=None \
        test_dataloader.dataset.ann_file="${VAL_ANN_FILE}" \
        test_dataloader.dataset.data_prefix.img="${IMAGES_PREFIX}" \
        test_dataloader.dataset.with_ref=False \
        ${CUSTOM_CLASSWISE} \
        test_evaluator.ann_file="${VAL_ANN_FILE}"
```


## Citation
If you find this work helpful please consider citing
```
@article{espinosa2024samantics,
        title={There is no SAMantics! Exploring SAM as a Backbone for Visual Understanding Tasks}, 
        author={Miguel Espinosa and Chenhongyi Yang and Linus Ericsson and Steven McDonagh and Elliot J. Crowley},
        year={2024},
        eprint={2405.00000},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }
