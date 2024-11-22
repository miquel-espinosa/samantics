import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# from projects.RefSAM.models.segment_anything.build_sam_baseline import sam_model_registry_baseline
from projects.RefSAM.models.segment_anything import build as sam_builder
from projects.RefSAM.models.segment_anything.predictor import SamPredictor


sam_checkpoint = "/work/scratch-nopw2/mespi/sam-hq-hf/sam-hq-training/pretrained_hq_sam/sam_hq_vit_b.pth"
model_type = "hq_sam_b"
device = "cuda"

# sam = sam_model_registry_baseline[model_type](checkpoint=sam_checkpoint)
sam = sam_builder.sam_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)



annotations_path = '/work/scratch-nopw2/mespi/sam-hq-hf/sam-hq-training/data/coco/annotations/instances_val2017.json'
bbox_prompts = 'work_dirs/RefSAM_det_coco_base_large/RefSAM_det_coco_base_large.bbox.json'


with open(bbox_prompts, 'r') as f:
    bboxes = json.load(f)

# Results in COCO format (list of dicts)
# E.g. [{'image_id': 42, 'category_id': 18, 'segmentation': [[x1, y1, x2, y2, ...]], 'score': 0.236}, ...]






eval_type = 'bbox'  # 'segm' for segmentation, 'bbox' for bounding box

coco = COCO(annotation_path)
coco_results = coco.loadRes(results_path)

# Evaluate
cocoEval = COCOeval(coco, coco_results, eval_type)
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()