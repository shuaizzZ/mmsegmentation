import torch
from torch import nn as nn

# 暂时放在这，以后要放到工具中
def int_size(x):
    size = []
    for i in x.size():
        size.append(int(i))
    return size

class DUpsamplingBlock(nn.Module):
    def __init__(self, inplanes, scale, num_class=21, pad=0):
        super(DUpsamplingBlock, self).__init__()
        self.inplanes = inplanes
        self.scale = scale
        self.num_class = num_class
        self.pad = pad
        ## W matrix
        self.conv_w = nn.Conv2d(inplanes, num_class * scale * scale, kernel_size=1, padding=pad, bias=False)
        self.T = torch.nn.Parameter(torch.Tensor([1.00]))  # softmax with temperature

    # 必须这么实现，否则self.conv_p 清除不掉的！！！
    def get_conv_p(self):
        ## TODO
        ## P matrix
        NSS = self.num_class * self.scale * self.scale
        conv_p = nn.Conv2d(NSS, self.inplanes, kernel_size=1, padding=self.pad, bias=False)
        return conv_p

    def mirror_process(self, mask):
        N, C, H, W = int_size(mask)
        C = self.num_class
        # N, C, H, W
        #mask = torch.unsqueeze(mask, dim=1)
        sample = torch.zeros(N, C, H, W).cuda()

        # 必须要把255这个标签去掉，否则下面scatter_会出错(但不在这里报错)
        mask[mask > C] = 0
        seggt_onehot = sample.scatter_(1, mask, 1)

        # N, H, W, C
        seggt_onehot = seggt_onehot.permute(0, 2, 3, 1)

        # N, H, W/sacle, C*scale
        WdC, CmS = int(W / self.scale), int(C * self.scale)
        seggt_onehot = seggt_onehot.contiguous()
        seggt_onehot = seggt_onehot.view((N, H, WdC, CmS))

        # N, W/sacle, H, C*scale
        seggt_onehot = seggt_onehot.permute(0, 2, 1, 3)

        # N, W/sacle, H/sacle, C*scale
        HdC, CmSS = int(H / self.scale), int(C * self.scale * self.scale)
        seggt_onehot = seggt_onehot.contiguous().view((N, WdC, HdC, CmSS))

        # N, C*scale*scale, H/sacle, W/sacle
        seggt_onehot = seggt_onehot.permute(0, 3, 2, 1)
        return seggt_onehot

    def forward_process(self, x):
        # N, C, H, W = x.size()
        N, C, H, W = int_size(x)

        ## N, W, H, C
        x_permuted = x.permute(0, 3, 2, 1)

        ## N, W, H*scale, C/scale
        HmC, CdS = int(H * self.scale), int(C / self.scale)
        x_permuted = x_permuted.contiguous().view((N, W, HmC, CdS))

        ## N, H*scale, W, C/scale
        x_permuted = x_permuted.permute(0, 2, 1, 3)

        ## N, H*scale, W*scale, C/(scale**2)
        WmC, CdSS = int(W * self.scale), int(C / (self.scale * self.scale))
        x_permuted = x_permuted.contiguous().view((N, HmC, WmC, CdSS))

        ## N, C/(scale**2), H*scale, W*scale
        x = x_permuted.permute(0, 3, 1, 2)
        return x

    def forward(self, x):
        x = self.conv_w(x)
        x = self.forward_process(x)
        x = x / self.T
        return x
