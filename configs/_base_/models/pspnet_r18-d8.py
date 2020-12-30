# ======================================= model settings ======================================= #
norm_cfg = dict(type='SyncBN', requires_grad=True)
dropout_ratio = 0.5
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
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=align_corners,
        dupsample=None,
        ppm_cfg=dict(ppm_channels=128,
                     pool_scales=(2, 4, 8, 16),
                     pooling='mix',
                     attention_cfg=None),
        loss_decode=[dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
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
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=align_corners,
        dupsample=None,
        loss_decode=[dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
                     ]
    ),
)
warmup_du_cfg = dict(
    interval=1,
    optimizer=dict(type='Ranger', lr=0.01, weight_decay=0.0005),
    # optimizer=dict(type='Adamax', lr=0.01, weight_decay=0.0005),
    total_runs=0,
    by_epoch=True)
# model training and testing settings
train_cfg = dict()
test_cfg = dict(mode='whole')