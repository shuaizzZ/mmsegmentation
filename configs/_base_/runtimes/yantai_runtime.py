# yapf:disable
log_config = dict(
   interval=8,
   hooks=[
       dict(type='StatisticTextLoggerHook', by_epoch=True),
       #dict(type='TensorboardLoggerHook')
   ])
# yapf:enable
seed = 4
deterministic = None
gpu_ids = [0]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
# workflow = [('train', 1), ('val', 1)]
workflow = [('train', 1)]
cudnn_benchmark = False
work_dir = './model'
