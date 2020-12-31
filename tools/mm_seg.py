# ======================================= model settings ======================================= #
norm_cfg = dict(type='BN', requires_grad=True)
num_classes = 2
dropout_ratio=0.1
# dupsample=dict(scale=8)
dupsample=None
# attention_cfg=dict(type='TPA')
attention_cfg=None
align_corners=False
model = dict(
    type='EncoderDecoder',
    pretrained='https://download.pytorch.org/models/resnet18-5c106cde.pth',
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
        dropout_ratio=dropout_ratio,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=align_corners,
        dupsample=dupsample,
        ppm_cfg=dict(ppm_channels=128,
                     pool_scales=(2, 4, 8, 16),
                     pooling='avg',
                     attention_cfg=attention_cfg),
        loss_decode=[dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                     dict(type='DiceLoss', loss_weight=1.0),
                     ]
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=2,
        channels=128,
        num_convs=1,
        concat_input=False,
        dropout_ratio=dropout_ratio,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=align_corners,
        dupsample=dupsample,
        loss_decode=[dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
                     dict(type='DiceLoss', loss_weight=0.4),
                     ]
    ),
)
warmup_du_cfg = dict(
    interval=10,
    optimizer=dict(type='SGD', lr=0.01),
    # optimizer=dict(type='Adamax', lr=0.01, weight_decay=0.0005),
    total_runs=0,
    by_epoch=False)
# model training and testing settings
train_cfg = dict()
test_cfg = dict(mode='whole')

# ======================================= dataset settings ======================================= #
data_root = '/root/public02/manuag/zhangshuai/data/ainno-example'
dataset_type = 'AinnoDataset'
dataset = 'example'
classes = ['background', 'abnormal']
labels = [0, 1]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale=(1024, 1024)
crop_size = (1024, 1024)
# img_scale=(256, 256)
# crop_size = (256, 256)

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
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Relabel', labels=labels),
            dict(type='MVCrop', crop_size=crop_size, crop_mode='center',
                 pad_mode=['constant', 'constant'], pad_fill=[0, 0], pad_expand=1.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'gt_semantic_seg']),
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

# ======================================= runtime settings ======================================= #
log_config = dict(
   interval=1,
   hooks=[
       dict(type='StatisticTextLoggerHook', by_epoch=True),
       #dict(type='TensorboardLoggerHook')
   ])
seed = 4
deterministic = None
cudnn_benchmark = False
gpu_ids = [0]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
# workflow = [('train', 1), ('val', 1)]
workflow = [('train', 1)]
work_dir = './model'

# ======================================= schedule settings ======================================= #
optimizer = dict(type='Ranger', lr=0.01, weight_decay=0.0005,
                 paramwise_cfg = dict(custom_keys={'backbone': dict(lr_mult=0.1)})) # , decay_mult=0.9
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.0)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=1e-3, by_epoch=True,
                 warmup='linear', warmup_iters=5, warmup_ratio=0.01,
                 warmup_by_epoch=True)
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
checkpoint_config = dict(by_epoch=True, interval=1, max_keep_ckpts=3)
evaluation = dict(interval=1, metric='mIoU', rescale=False, ignore_index=[0],
                  f1_cfg=dict(com_f1=True, type='pix_iof', threshold=[0, 0.3],
                              defect_filter=dict(type='box', size=[10, 10])),
                  best_metrics = ['IoU', 'Acc', 'Recall', 'Precision', 'F1'])
# TODO load best_metrics when resume

# ======================================= convert ======================================= #
convert_size = (1024, 1024)
