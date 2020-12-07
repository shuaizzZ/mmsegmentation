# ======================================= model settings ======================================= #
norm_cfg = dict(type='BN', requires_grad=True)
num_classes = 2
# dupsample=dict(scale=8)
dupsample=None
align_corners=False
pretrained_name='resnet18'
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True
    ),
    decode_head=dict(
        type='PSPHead',
        in_channels=512,
        in_index=3,
        channels=128,
        pool_scales=(2, 4, 8, 16),
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=align_corners,
        dupsample=dupsample,
        pooling='mix',
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=2,
        channels=128,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=align_corners,
        dupsample=dupsample,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
    ),
)
du_config = dict(
    interval=10,
    optimizer=dict(type='Adamax', lr=0.01, weight_decay=0.0005),
    total_runs=200,
    by_epoch=False)
# model training and testing settings
train_cfg = dict()
test_cfg = dict(mode='whole')

# ======================================= dataset settings ======================================= #
data_root = '/root/public02/manuag/zhangshuai/data/cicai_data/cicai-hangzhou'
dataset_type = 'AinnoDataset'
dataset = 'example'
classes = ['background', '1diaojiao', '2liewen', '3kongdong']
#INDEXES: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#LABELS = [0, 1, 1, 1, 1, 1, 0, 1, 1, 0] # 2
#LABELS = [0, 1, 2, 0, 0, 0, 0, 2, 0, 0] # 3
#LABELS = [0, 1, 2, 3, 3, 0, 0, 2, 3, 0] # 4
labels = [0, 1, 2, 3, 3, 0, 0, 2, 0, 1] # 4

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale=(1024, 1024)
crop_size = (1024, 1024)
# img_scale=(512, 512)
# crop_size = (512, 512)

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
            dict(type='MVCrop', crop_size=crop_size, crop_mode='center',
                 pad_mode=['range', 'constant'], pad_fill=[[0, 255], 0], pad_expand=1.0),
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
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        dataset=dataset,
        data_root=data_root,
        classes=classes,
        split='test',
        test_mode=True,
        pipeline=test_pipeline),)

# ======================================= runtime settings ======================================= #
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        # dict(type='TensorboardLoggerHook')
    ])
seed = 4
deterministic = None
gpu_ids = [1]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
# workflow = [('train', 1), ('val', 1)]
workflow = [('train', 1)]
cudnn_benchmark = False
work_dir = './model'

# ======================================= schedule settings ======================================= #
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer = dict(type='Adamax', lr=0.01, weight_decay=0.0005)
paramwise_cfg = dict(custom_keys={'.backbone': dict(lr_mult=0.1, decay_mult=0.9)})
optimizer_config = dict()
# learning policy
# lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
lr_config = dict(policy='CosineAnnealing', min_lr=1e-4, by_epoch=True,
                 warmup='linear', warmup_iters=6, warmup_ratio=0.01,
                 warmup_by_epoch=True)
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)
checkpoint_config = dict(by_epoch=True, interval=1, max_keep_ckpts=3)
evaluation = dict(interval=1, metric='mIoU')

