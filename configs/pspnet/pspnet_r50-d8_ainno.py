_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/ainno_dataset.py',
    '../_base_/ainno_runtime.py', '../_base_/schedules/schedule_ainno.py'
]
model = dict(
    decode_head=dict(num_classes=4), auxiliary_head=dict(num_classes=4))
test_cfg = dict(mode='whole')
