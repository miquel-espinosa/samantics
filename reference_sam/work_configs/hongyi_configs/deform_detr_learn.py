_base_ = '../configs/deformable_detr/deformable-detr_r50_16xb2-50e_coco.py'
model = dict(with_box_refine=True)
