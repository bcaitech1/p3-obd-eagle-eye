# dataset settings
dataset_type = "CocoDataset"
data_root = "/opt/ml/input/data/"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

# train and test dataset pipeline
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="Resize", img_scale=(512, 512), keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

classes = (
    "UNKNOWN",
    "General trash",
    "Paper",
    "Paper pack",
    "Metal",
    "Glass",
    "Plastic",
    "Styrofoam",
    "Plastic bag",
    "Battery",
    "Clothing",
)


# create custom dataset
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + "train.json",
        img_prefix=data_root,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + "val.json",
        img_prefix=data_root,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + "test.json",
        img_prefix=data_root,
        pipeline=test_pipeline,
    ),
)
evaluation = dict(metric=["bbox"])

# model settings
model = dict(
    type="VFNet",
    pretrained="open-mmlab://resnext101_32x4d",
    backbone=dict(
        type="ResNeXt",
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=True,
        style="pytorch",
        dcn=dict(type="DCNv2", deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
    ),
    neck=dict(
        type="FPN",
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        extra_convs_on_inputs=False,  # use P5
        num_outs=5,
        relu_before_extra_convs=True,
    ),
    bbox_head=dict(
        type="VFNetHead",
        num_classes=11,
        in_channels=256,
        stacked_convs=3,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=False,
        dcn_on_last_conv=True,
        use_atss=True,
        use_vfl=True,
        loss_cls=dict(
            type="VarifocalLoss",
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0,
        ),
        loss_bbox=dict(type="GIoULoss", loss_weight=1.5),
        loss_bbox_refine=dict(type="GIoULoss", loss_weight=2.0),
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type="ATSSAssigner", topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type="nms", iou_threshold=0.6),
        max_per_img=100,
    ),
)

# pretrained
load_from = "https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/vfnet/vfnet_x101_32x4d_fpn_mdconv_c3-c5_mstrain_2x_coco/vfnet_x101_32x4d_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-d300a6fc.pth"

# optimizer
optimizer = dict(
    type="SGD",
    lr=0.002,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(bias_lr_mult=2.0, bias_decay_mult=0.0),
)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.1, step=[16, 22]
)
runner = dict(type="EpochBasedRunner", max_epochs=30)

# runtime settings
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="trash-recycling-object-detection", name="resnext101_fpn_vfnet"
            ),
        ),
    ],
)

custom_hooks = [dict(type="NumClassCheckHook")]

dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]
