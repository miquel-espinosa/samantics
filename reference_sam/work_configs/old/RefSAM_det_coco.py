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

sam_vit_c = 160
num_feature_levels = 1

model = dict(
    type='RefSAMDetector',
    num_queries=300,
    embed_dims=256,
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
        type="hq_sam_t",
        # checkpoint="/localdisk/data2/hongyi/ckpts/sam_hq_download/sam_hq_vit_tiny.pth"
        checkpoint="/work/scratch-nopw2/mespi/sam-hq-hf/sam-hq-training/pretrained_hq_sam/sam_hq_vit_tiny.pth"
    ),
    neck=dict(
        type='ChannelMapper',
        in_channels=[160],
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
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=100)
)



dataset_type = 'COCORefSegTrainDataset'
data_root = "/localdisk/data1/public_datasets/coco"
data_root = "/work/scratch-nopw2/mespi/sam-hq-hf/sam-hq-training/data/coco/images"

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotationsCorrect', with_bbox=True, with_mask=True),
    dict(type='FixShapeResize', width=1024, height=1024, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='FixShapeResize', width=1024, height=1024, keep_ratio=True),
    dict(type='LoadAnnotationsCorrect', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # ann_file="/localdisk/data1/public_datasets/coco/annotations/instances_train2017.json",
        ann_file="/work/scratch-nopw2/mespi/sam-hq-hf/sam-hq-training/data/coco/annotations/instances_train2017.json",
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline,
        backend_args=backend_args)
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # ann_file="/localdisk/data1/public_datasets/coco/annotations/instances_val2017.json",
        ann_file="/work/scratch-nopw2/mespi/sam-hq-hf/sam-hq-training/data/coco/annotations/instances_val2017.json",
        data_prefix=dict(img='val2017/'),
        test_mode=False,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    # ann_file='/localdisk/data1/public_datasets/coco/annotations/instances_val2017.json',
    ann_file='/work/scratch-nopw2/mespi/sam-hq-hf/sam-hq-training/data/coco/annotations/instances_val2017.json',
    metric=['bbox'],
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }
    )
)

# learning policy
max_epochs = 50
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=9999999
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[40],
        gamma=0.1)
]
auto_scale_lr = dict(base_batch_size=32)
