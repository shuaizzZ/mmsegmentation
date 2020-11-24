_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/yantai_st12.py',
    '../_base_/yantai_runtime.py', '../_base_/schedules/schedule_yantai.py'
]
model = dict(
    decode_head=dict(num_classes=4), auxiliary_head=dict(num_classes=4))
test_cfg = dict(mode='whole')
