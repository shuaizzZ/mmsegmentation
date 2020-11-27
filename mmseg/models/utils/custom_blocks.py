
import torch
import torch.nn as nn

class Mix2Pooling(nn.Module):
    def __init__(self, size):
        super(Mix2Pooling, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(size)
        self.max_pool = nn.AdaptiveMaxPool2d(size)

    def forward(self, x):
        spx = torch.chunk(x, 2, 1)
        out = torch.cat((self.avg_pool(spx[0]), self.max_pool(spx[1])), 1)
        return out