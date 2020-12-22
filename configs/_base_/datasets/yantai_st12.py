# dataset settings
dataset_type = 'YantaiDataset' # YantaiDataset
data_root = '/opt/data/public02/manuag/zhangshuai/data'
dataset = "yantai-12_v2345_unq_1008" # 12_v234_all_1c  12_v234_unq_1c

classes = ['background', '1diaojiao', '2liewen', '3kongdong']
#INDEXES: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#LABELS = [0, 1, 1, 1, 1, 1, 0, 1, 1, 0] # 2
#LABELS = [0, 1, 2, 0, 0, 0, 0, 2, 0, 0] # 3
#LABELS = [0, 1, 2, 3, 3, 0, 0, 2, 3, 0] # 4
labels = [0, 1, 2, 3, 3, 0, 0, 2, 0, 1] # 4

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale=(512, 512)
crop_size = (512, 512)
# crop_size = (128, 128)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='XYShift', shift=(6, 3)),
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
            dict(type='MVCrop', crop_size=crop_size, crop_mode='center',
                 pad_mode=['constant', 'constant'], pad_fill=[0, 0], pad_expand=1.0),
            dict(type='XYShift', shift=(1, 1)),
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
    samples_per_gpu=32,
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