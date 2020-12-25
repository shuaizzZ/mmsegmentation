
import torch
from torch import nn as nn
from .custom_blocks import int_size
from ..builder import build_loss
from mmcv.runner import auto_fp16, force_fp32


class DUpsamplingBlock(nn.Module):
    def __init__(self, inplanes, scale, num_class=21, pad=0):
        super(DUpsamplingBlock, self).__init__()
        self.inplanes = inplanes
        self.scale = scale
        self.num_class = num_class
        self.pad = pad
        ## W matrix
        NSS = self.num_class * self.scale * self.scale
        self.conv_w = nn.Conv2d(inplanes, NSS, kernel_size=1, padding=pad, bias=False)
        self.T = torch.nn.Parameter(torch.Tensor([1.00]))  # softmax with temperature

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


class MirrorDUpsamplingBlock(nn.Module):
    def __init__(self, du_block, loss_cfg=dict(type='MSELoss')):
        super(MirrorDUpsamplingBlock, self).__init__()
        self.fp16_enabled = False
        self.inplanes = du_block.inplanes
        self.scale = du_block.scale
        self.num_class = du_block.num_class
        self.pad = du_block.pad
        self.conv_w = du_block.conv_w
        ## P matrix
        NSS = self.num_class * self.scale * self.scale
        self.conv_p = nn.Conv2d(NSS, self.inplanes, kernel_size=1, padding=self.pad, bias=False)
        self.loss_du = build_loss(loss_cfg)

    def mirror_process(self, mask):
        N, _, H, W = int_size(mask)  # N, 1, H, W
        C = self.num_class

        # N, C, H, W
        sample = torch.zeros(N, C, H, W)#.cuda(mask.device.index)

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
        seggt_onehot = seggt_onehot.permute(0, 3, 2, 1).float()
        return seggt_onehot

    @auto_fp16()
    def forward(self, seggt_onehot):
        seggt_onehot_reconstructed = self.forward_train(seggt_onehot)
        loss = self.losses(seggt_onehot, seggt_onehot_reconstructed)
        return loss

    @auto_fp16()
    def forward_train(self, seggt_onehot):
        seggt_onehot_reconstructed = self.conv_p(seggt_onehot)
        seggt_onehot_reconstructed = self.conv_w(seggt_onehot_reconstructed)
        return seggt_onehot_reconstructed

    @force_fp32()
    def losses(self, seggt_onehot, seggt_onehot_reconstructed):
        loss = self.loss_du(seggt_onehot, seggt_onehot_reconstructed)
        if loss.item() == 0:
            print()
        return loss
