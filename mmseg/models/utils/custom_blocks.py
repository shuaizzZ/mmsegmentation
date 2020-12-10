
import torch
import torch.nn as nn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

def int_size(x):
    size = tuple(int(s) for s in x.size())
    return size

def parallel_model(model, gpu_ids, distributed, find_unused_parameters=False):
    # put model on gpus
    if distributed:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(gpu_ids[0]), device_ids=gpu_ids)
    return model

class Mix2Pooling(nn.Module):
    def __init__(self, size):
        super(Mix2Pooling, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(size)
        self.max_pool = nn.AdaptiveMaxPool2d(size)

    def forward(self, x):
        spx = torch.chunk(x, 2, 1)
        out = torch.cat((self.avg_pool(spx[0]), self.max_pool(spx[1])), 1)
        return out