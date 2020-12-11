_base_ = [
    '../_base_/models/du_pspnet_r50-d8.py', '../_base_/datasets/yantai_st12.py',
    '../_base_/yantai_runtime.py', '../_base_/schedules/schedule_yantai.py'
]
model = dict(
    decode_head=dict(num_classes=4), auxiliary_head=dict(num_classes=4))
test_cfg = dict(mode='whole')

warmup_du_cfg = dict(
    interval=10,
    optimizer=dict(type='SGD', lr=0.01),
    total_runs=40,
    by_epoch=False
)