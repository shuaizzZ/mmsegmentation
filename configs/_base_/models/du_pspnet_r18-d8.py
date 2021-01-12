# ======================================= model settings ======================================= #
norm_cfg = dict(type='SyncBN', requires_grad=True)
dropout_ratio = 0.5 # When using BN, this parameter has little effect on the result
dupsample=dict(scale=8)
# dupsample=None
attention_cfg=dict(type='CBAM', ratio=16, kernel_size=7)
# attention_cfg=None
sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=int(512*512*0.5))
align_corners=False
model = dict(
    type='EncoderDecoder',
    pretrained='https://download.pytorch.org/models/resnet18-5c106cde.pth',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
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
        channels=256,
        dropout_ratio=dropout_ratio,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=align_corners,
        dupsample=dupsample,
        ppm_cfg=dict(ppm_channels=128,
                     pool_scales=(2, 4, 8, 16),
                     pooling='mix',
                     attention_cfg=attention_cfg,),
        loss_decode=[dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                     dict(type='DiceLoss', loss_weight=1.0),
                     ]
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=512,
        in_index=3,
        channels=128,
        num_convs=1,
        concat_input=False,
        dropout_ratio=dropout_ratio,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=align_corners,
        dupsample=dupsample,
        loss_decode=[dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
                     dict(type='DiceLoss', loss_weight=0.4),
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