_base_ = [
    '../_base_/models/du_pspnet_r18-d8.py', '../_base_/datasets/ainno_dataset.py',
    '../_base_/runtimes/ainno_runtime.py', '../_base_/schedules/schedule_ainno.py'
]
num_classes = 2
model = dict(
    decode_head=dict(num_classes=num_classes), auxiliary_head=dict(num_classes=num_classes))
test_cfg = dict(mode='whole')