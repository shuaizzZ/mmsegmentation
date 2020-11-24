# dataset settings
dataset_type = 'YantaiDataset'
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

crop_size = (512, 768)
size = (512, 768)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Relabel', labels=labels),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=size, pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Relabel', labels=labels),
    dict(type='Pad', size=size, pad_val=0, seg_pad_val=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'],
         meta_keys=('filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor',
                    'flip_direction', 'img_norm_cfg')),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
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
        split='train',
        test_mode=False,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        dataset=dataset,
        data_root=data_root,
        classes=classes,
        split='val',
        test_mode=False,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        dataset=dataset,
        data_root=data_root,
        classes=classes,
        split='val',
        test_mode=True,
        pipeline=test_pipeline),)
