# dataset settings
data_root = '/root/public02/manuag/zhangshuai/data/ainno-example'
dataset_type = 'AinnoDataset'
dataset = 'example'
classes = ['background', 'abnormal']
labels = [0, 1]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale=(1024, 1024)
crop_size = (1024, 1024)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Relabel', labels=labels),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='MVResize', h_range=(0.8, 1.2), w_range=(0.8, 1.2), keep_ratio=True),
    dict(type='MVRotate', angle_range=(-45, 45), center=None),
    dict(type='MVCrop', crop_size=crop_size, crop_mode='random',
         pad_mode=["range", "constant"], pad_fill=[[0, 255], 0], pad_expand=1.2),
    dict(type='PhotoMetricDistortion',
         brightness_delta=50,
         contrast_range=(0.8, 1.2),
         saturation_range=(0.8, 1.2),
         hue_delta=50),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            # dict(type='MVCrop', crop_size=crop_size, crop_mode='center',
            #      pad_mode=['range', 'constant'], pad_fill=[[0, 255], 0], pad_expand=1.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        dataset=dataset,
        data_root=data_root,
        classes=classes,
        labels=labels,
        split='train',
        test_mode=False,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        dataset=dataset,
        data_root=data_root,
        classes=classes,
        labels=labels,
        split='val',
        test_mode=False,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        dataset=dataset,
        data_root=data_root,
        classes=classes,
        split='test',
        test_mode=True,
        pipeline=test_pipeline),)