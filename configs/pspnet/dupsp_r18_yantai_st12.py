_base_ = [
    '../_base_/models/du_pspnet_r18-d8.py', '../_base_/datasets/yantai_st12.py',
    '../_base_/runtimes/yantai_runtime.py', '../_base_/schedules/schedule_yantai.py'
]
num_classes = 4
model = dict(
    decode_head=dict(num_classes=num_classes), auxiliary_head=dict(num_classes=num_classes))
test_cfg = dict(mode='whole')