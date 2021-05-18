_base_ = '/opt/ml/code/mmdetection_trash/configs/cascade_rcnn/cascade_mask_rcnn_x101_32x4d_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))