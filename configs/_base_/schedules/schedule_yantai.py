# optimizer
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer = dict(type='Ranger', lr=0.01, weight_decay=0.0005)
paramwise_cfg = dict(custom_keys={'.backbone': dict(lr_mult=0.1, decay_mult=0.9)})
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.0)
# learning policy
# lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
lr_config = dict(policy='CosineAnnealing', min_lr=1e-4, by_epoch=True,
                 warmup='linear', warmup_iters=8, warmup_ratio=0.01,
                 warmup_by_epoch=True)
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=1500)
checkpoint_config = dict(by_epoch=True, interval=1, max_keep_ckpts=3, best_type='sum')
evaluation = dict(interval=1, metric='mIoU', rescale=False,
                  ignore_index=[0], com_f1=True,
                  defect_metric=dict(TYPE='pix_iof', THRESHOLD=[0, 0.2, 0.2, 0.2]),
                  defect_filter=dict(TYPE='', SIZE_ALL=[16, 16]),
                  best_metrics = ['IoU', 'Acc', 'Recall', 'Precision', 'F1'])
