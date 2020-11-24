# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
paramwise_cfg = dict(custom_keys={'.backbone': dict(lr_mult=0.1, decay_mult=0.9)})
optimizer_config = dict()
# learning policy
# lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=True,
                 warmup='linear', warmup_iters=600, warmup_ratio=0.1,
                 warmup_by_epoch=False)
# runtime settings
# runner = dict(type='IterBasedRunner', max_iters=8000)
runner = dict(type='EpochBasedRunner', max_epochs=800)
checkpoint_config = dict(by_epoch=True, interval=1)
evaluation = dict(interval=1, metric='mIoU')
