import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS, build_attention
from .decode_head import BaseDecodeHead
from ..utils import Mix2Pooling


class PPM(nn.Module):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, ppm_channels, conv_cfg, norm_cfg,
                 act_cfg, align_corners, pooling='avg', attention_cfg=None):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.ppm_channels = ppm_channels
        self.out_channels = in_channels + len(pool_scales) * ppm_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        if pooling == 'avg':
            self.pooling = nn.AdaptiveAvgPool2d
        elif pooling == 'mix':
            self.pooling = Mix2Pooling
        self.ppms = nn.ModuleList()
        for pool_scale in pool_scales:
            self.ppms.append(
                nn.Sequential(
                    self.pooling(pool_scale),
                    ConvModule(
                        self.in_channels,
                        self.ppm_channels,
                        1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg)))
        if attention_cfg:
            attention_cfg['in_channels'] = self.out_channels
            self.attention = build_attention(attention_cfg)

    def forward(self, x):
        """Forward function."""
        ppm_outs = [x]
        for ppm in self.ppms:
            ppm_out = ppm(x)
            upsampled_ppm_out = resize(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        ppm_outs = torch.cat(ppm_outs, dim=1)
        if hasattr(self, 'attention'):
            ppm_outs = self.attention(ppm_outs)
        return ppm_outs


@HEADS.register_module()
class PSPHead(BaseDecodeHead):
    """Pyramid Scene Parsing Network.

    This head is the implementation of
    `PSPNet <https://arxiv.org/abs/1612.01105>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module. Default: (1, 2, 3, 6).
    """

    def __init__(self,
                 pool_scales=(1, 2, 3, 6),
                 ppm_channels=128,
                 pooling='avg',
                 attention_cfg=None,
                 **kwargs):
        super(PSPHead, self).__init__(**kwargs)
        assert isinstance(pool_scales, (list, tuple))
        self.pool_scales = pool_scales
        self.ppm = PPM(
            self.pool_scales,
            self.in_channels,
            ppm_channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners,
            pooling=pooling,
            attention_cfg=attention_cfg)
        self.bottleneck = ConvModule(
            self.ppm.out_channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        ppm_outs = self.ppm(x)
        output = self.bottleneck(ppm_outs)
        output = self.cls_seg(output)
        return output
