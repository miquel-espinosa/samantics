_base_ = [
    '../configs/_base_/default_runtime.py'
]

custom_imports = dict(
    imports=[
        'projects.RefSAM.models',
        'projects.RefSAM.datasets'
    ],
    allow_failed_imports=False
)

sam_vit_c = 768
num_feature_levels = 4

model = dict(
    type='RefSAMDetector',
    num_queries=300,
    embed_dims=256,
    precompute_n_refs=10, # Precomputed embeddings for 10 reference images per class
    # If path specified, load precomputed reference embeddings from this path and run forward method with target imgs
    # load_precompute_ref_path='/localdisk/data2/Users/s2254242/datasets/coco/precompute_10_shot_annotations_ref_train.npy',
    load_precompute_ref_path='/work/scratch-nopw2/mespi/sam-hq-hf/sam-hq-training/data/coco/precompute_10_shot_annotations_ref_by_area.npy',
    n_shot=10,
    num_feature_levels=num_feature_levels,
    with_box_refine=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=None,
        std=None,
        bgr_to_rgb=False,
        pad_size_divisor=1
    ),
    sam=dict(
        type="hq_sam_b",
        # checkpoint="/localdisk/data2/Users/s2254242/projects_storage/ref_sam/hq-sam/sam_hq_vit_b.pth"
        checkpoint="/work/scratch-nopw2/mespi/sam-hq-hf/sam-hq-training/pretrained_hq_sam/sam_hq_vit_b.pth"
    ),
    neck=dict(
        type='ChannelMapper',
        in_channels=[768, 768, 768, 768],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=num_feature_levels
    ),
    encoder=dict(  # DeformableDetrTransformerEncoder
        num_layers=1,
        layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
            self_attn_cfg=dict(  # MultiScaleDeformableAttention
                embed_dims=256,
                batch_first=True
            ),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=256 * 2,
                ffn_drop=0.1
            )
        )
    ),
    decoder=dict(  # DeformableDetrTransformerDecoder
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(  # DeformableDetrTransformerDecoderLayer
            self_attn_cfg=dict(  # MultiheadAttention
                embed_dims=256,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            ),
            cross_attn_cfg=dict(  # MultiScaleDeformableAttention
                embed_dims=256,
                batch_first=True
            ),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=1024,
                ffn_drop=0.1
            )
        ),
        post_norm_cfg=None
    ),
    positional_encoding=dict(num_feats=128, normalize=True, offset=-0.5),
    bbox_head=dict(
        type='DeformableDETRHead',
        num_classes=1,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    test_cfg=dict(max_per_img=100)
)



dataset_type = 'COCORefSegTrainDataset'
# data_root = "/localdisk/data1/Data/COCO"
data_root = "/work/scratch-nopw2/mespi/sam-hq-hf/sam-hq-training/data/coco"

backend_args = None


test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotationsCorrect', with_bbox=True, with_mask=True),
    dict(type='FixShapeResize', width=1024, height=1024, keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]


val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        # We only care about target images, the ref are discarded and loaded the precomputed ones
        ref_equal_target=False,
        data_root=data_root,
        # ann_file="/localdisk/data1/Data/COCO/annotations_trainval2017/annotations/instances_val2017.json",
        ann_file="/work/scratch-nopw2/mespi/sam-hq-hf/sam-hq-training/data/coco/annotations/instances_val2017.json",
        # data_prefix=dict(img='val2017/val2017/'),
        data_prefix=dict(img='val2017/'),
        filter_cfg=dict(filter_empty_gt=True),
        test_mode=False,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    # ann_file='/localdisk/data1/Data/COCO/annotations_trainval2017/annotations/instances_val2017.json',
    ann_file="/work/scratch-nopw2/mespi/sam-hq-hf/sam-hq-training/data/coco/annotations/instances_val2017.json",
    metric=['bbox'],
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

auto_scale_lr = dict(base_batch_size=32)

vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')