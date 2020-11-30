_base_ = [
    '../_base_/models/du_pspnet_r50-d8.py', '../_base_/datasets/yantai_st12.py',
    '../_base_/yantai_runtime.py', '../_base_/schedules/schedule_yantai.py'
]
model = dict(
    decode_head=dict(num_classes=4), auxiliary_head=dict(num_classes=4))
test_cfg = dict(mode='whole')

du_config = dict(
    interval=200,
    optimizer=dict(type='SGD', lr=0.01),
    total_runs=1000,
    by_epoch=False
)