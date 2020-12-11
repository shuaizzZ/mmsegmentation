_base_ = [
    '../_base_/models/du_pspnet_r50-d8.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    decode_head=dict(num_classes=150), auxiliary_head=dict(num_classes=150))
test_cfg = dict(mode='whole')

warmup_du_cfg = dict(
    interval=200,
    optimizer=dict(type='SGD', lr=0.002),
    total_runs=1000,
    by_epoch=False
)
