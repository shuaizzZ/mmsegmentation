# yapf:disable
log_config = dict(
   interval=1,
   hooks=[
       dict(type='StatisticTextLoggerHook', by_epoch=True, interval=1),
       #dict(type='TensorboardLoggerHook')
   ])
# yapf:enable
seed = None
deterministic = False
gpu_ids = [0]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
# workflow = [('train', 1), ('val', 1)]
workflow = [('train', 1)]
cudnn_benchmark = False
work_dir = '/root/public02/manuag/zhangshuai/data/yantai_results'

segmentor_type = 'manuvision'
task_name = '2021_0104_17_r18_nodilate_dice_p^1'
