norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='PSPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=4,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        dupsample=dict(scale=8),
        pooling='mix',
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=4,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        dupsample=dict(scale=8),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)))
train_cfg = dict()
test_cfg = dict(mode='whole')
dataset_type = 'YantaiDataset'
data_root = '/opt/data/public02/manuag/zhangshuai/data'
dataset = 'yantai-12_v2345_unq_1008'
classes = ['background', '1diaojiao', '2liewen', '3kongdong']
labels = [0, 1, 2, 3, 3, 0, 0, 2, 0, 1]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale = (512, 512)
crop_size = (512, 512)
size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Relabel', labels=[0, 1, 2, 3, 3, 0, 0, 2, 0, 1]),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=12,
    workers_per_gpu=4,
    train=dict(
        type='YantaiDataset',
        dataset='yantai-12_v2345_unq_1008',
        data_root='/opt/data/public02/manuag/zhangshuai/data',
        classes=['background', '1diaojiao', '2liewen', '3kongdong'],
        split='train',
        test_mode=False,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(type='Relabel', labels=[0, 1, 2, 3, 3, 0, 0, 2, 0, 1]),
            dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=0),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='YantaiDataset',
        dataset='yantai-12_v2345_unq_1008',
        data_root='/opt/data/public02/manuag/zhangshuai/data',
        classes=['background', '1diaojiao', '2liewen', '3kongdong'],
        split='val',
        test_mode=False,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='YantaiDataset',
        dataset='yantai-12_v2345_unq_1008',
        data_root='/opt/data/public02/manuag/zhangshuai/data',
        classes=['background', '1diaojiao', '2liewen', '3kongdong'],
        split='val',
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=1, hooks=[dict(type='TextLoggerHook', by_epoch=True)])
seed = 4
deterministic = None
gpu_ids = [0, 1, 2, 3]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = False
optimizer = dict(type='Adamax', lr=0.01, weight_decay=0.0005)
paramwise_cfg = dict(
    custom_keys=dict({'.backbone': dict(lr_mult=0.1, decay_mult=0.9)}))
optimizer_config = dict()
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.0001,
    by_epoch=True,
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.01,
    warmup_by_epoch=False)
runner = dict(type='EpochBasedRunner', max_epochs=2400)
checkpoint_config = dict(by_epoch=True, interval=1, max_keep_ckpts=10)
evaluation = dict(interval=1, metric='mIoU')
du_config = dict(
    interval=200,
    optimizer=dict(type='SGD', lr=0.01),
    total_runs=1000,
    by_epoch=False)
work_dir = '../work_dir/base_1127_dup2p/'