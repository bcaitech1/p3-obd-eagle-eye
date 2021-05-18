# wandb
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='pstage-3-2-detection',
                name='DCN_cascade_mask_rcnn_x101_32x4d_fpn_1x_coco_pretrained_basic' # run_name
        )
    )
])

# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/opt/ml/pretrained_models/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco-e75f90c8.pth'
resume_from = None
workflow = [('train', 1)]
