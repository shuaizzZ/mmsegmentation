
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from mmcv.utils import Registry, build_from_cfg
from .custom_blocks import Mix2Pooling, int_size
from mmseg.models.builder import ATTENTION


# ---------------------------------------------------------------------------- #
# SELayer
# ---------------------------------------------------------------------------- #
@ATTENTION.register_module()
class SE(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        inter_channels = in_channels // reduction
        self.fc = nn.Sequential(
            nn.Linear(in_channels, inter_channels, bias=False),
            nn.ReLU(inplace=True),
            # Mish(),
            nn.Linear(inter_channels, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = int_size(x)

        y = self.avg_pool(x).view(n, c)
        y_expand = self.fc(y).view(n, c, 1, 1)

        return x * y_expand

# ---------------------------------------------------------------------------- #
# NonLocal Block
# ---------------------------------------------------------------------------- #
@ATTENTION.register_module()
class NonLocal(nn.Module):
    def __init__(self, in_channels):
        super(NonLocal, self).__init__()
        self.inter_channel = in_channels // 2
        self.conv_phi = nn.Conv2d(in_channels=in_channels, out_channels=self.inter_channel,
                                  kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=in_channels, out_channels=self.inter_channel,
                                    kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=in_channels, out_channels=self.inter_channel,
                                kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=in_channels,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = int_size(x)
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, c, -1)
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0,2,1).contiguous().view(b,self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out

# ---------------------------------------------------------------------------- #
# SCSE Block
# ---------------------------------------------------------------------------- #
class SpatialSE(nn.Module):
    def __init__(self, channel):
        super(SpatialSE, self).__init__()
        self.spatial_excitation = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self,x):
        spatial_weirht = self.spatial_excitation(x)
        return x * spatial_weirht

class ChannelSE(nn.Module):
    def __init__(self, channel, reduction=4):
        super(ChannelSE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduction_channel = int(channel / reduction)
        self.fc = nn.Sequential(
            nn.Linear(channel, reduction_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduction_channel, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c = int(x.size(0)), int(x.size(1))

        y = self.avg_pool(x).view(n, c)
        y_expand = self.fc(y).view(n, c, 1, 1)

        return x * y_expand

@ATTENTION.register_module()
class SCSE(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SCSE, self).__init__()
        self.spatialSE = SpatialSE(in_channels)
        self.channelSE = ChannelSE(in_channels, reduction=reduction)

    def forward(self,x):
        return self.spatialSE(x) + self.channelSE(x)

@ATTENTION.register_module()
class SCSE2(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SCSE2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduction_channel = int(in_channels / reduction)
        self.channel_excitation = nn.Sequential(
            nn.Linear(in_channels, reduction_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduction_channel, in_channels, bias=False),
            nn.Sigmoid()
        )
        self.spatial_excitation = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self,x):
        n, c = int(x.size(0)), int(x.size(1))

        y = self.avg_pool(x).view(n, c)
        channel_weirht = self.channel_excitation(y).view(n, c, 1, 1)
        spatial_weirht = self.spatial_excitation(x)
        x = x * channel_weirht * spatial_weirht
        return x

# ---------------------------------------------------------------------------- #
# CBAM :: Convolutional Block Attention Module
# ---------------------------------------------------------------------------- #
## ---- 通道Attention ---- ##
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, int(in_planes / ratio), 1, bias=False), nn.ReLU(),
            nn.Conv2d(int(in_planes / ratio), in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return x * self.sigmoid(avgout + maxout)

class ChannelAttentionMixPooling(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttentionMixPooling, self).__init__()
        self.mix_pool = Mix2Pooling(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, int(in_planes / ratio), 1, bias=False), nn.ReLU(),
            nn.Conv2d(int(in_planes / ratio), in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        mixout = self.sharedMLP(self.mix_pool(x))
        return x * self.sigmoid(mixout)

## ---- 空间Attention ---- ##
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)

@ATTENTION.register_module()
class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ChannelAttention = ChannelAttention(in_channels, ratio)
        self.SpatialAttention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ChannelAttention(x)
        x = self.SpatialAttention(x)
        return x

@ATTENTION.register_module()
class MCBAM(nn.Module):
    def __init__(self, in_channels, ratio=16, kernel_size=7):
        super(MCBAM, self).__init__()
        self.ChannelAttention = ChannelAttentionMixPooling(in_channels, ratio)
        self.SpatialAttention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ChannelAttention(x)
        x = self.SpatialAttention(x)
        return x

# ---------------------------------------------------------------------------- #
# EMANet :: https://github.com/XiaLiPKU/EMANet
# ---------------------------------------------------------------------------- #
class EMAU_ORG(nn.Module):
    '''The Expectation-Maximization Attention Unit (EMAU).

    Arguments:
        c (int): The input and output channel number.
        k (int): The number of the bases.
        stage_num (int): The iteration number for EM.
    '''
    def __init__(self, channel, k, stage_num=3, norm_layer=nn.BatchNorm2d):
        super(EMAU_ORG, self).__init__()
        self.stage_num = stage_num

        mu = torch.Tensor(1, channel, k)
        mu.normal_(0, math.sqrt(2. / k))  # Init with Kaiming Norm.
        mu = self._l2norm(mu, dim=1)
        self.register_buffer('mu', mu)
        self.momentum = 0.9

        self.conv1 = nn.Conv2d(channel, channel, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel, channel, 1, bias=False),
            norm_layer(channel))
        self.relu = nn.ReLU(True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _l2norm(self, inp, dim):
        '''Normlize the inp tensor with l2-norm.

        Returns a tensor where each sub-tensor of input along the given dim is
        normalized such that the 2-norm of the sub-tensor is equal to 1.

        Arguments:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.

        Returns:
            (tensor) The normalized tensor.
        '''
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

    def forward(self, x):
        idn = x
        # The first 1x1 conv
        x = self.conv1(x)

        # The EM Attention
        b, c, h, w = int_size(x)
        x = x.view(b, c, h * w)  # b * c * n
        mu = self.mu.repeat(b, 1, 1)  # 1 * c * k ==> b * c * k

        with torch.no_grad():
            for i in range(self.stage_num):
                x_t = x.permute(0, 2, 1).contiguous()  # b * n * c
                z = torch.bmm(x_t, mu)  # b * n * k
                z = F.softmax(z, dim=2)  # b * n * k
                z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
                mu = torch.bmm(x, z_)  # b * c * k
                mu = self._l2norm(mu, dim=1)

        # !!! The moving averaging operation
            if self.training:
                mmu = mu.mean(dim=0, keepdim=True)
                self.mu *= self.momentum
                self.mu += mmu * (1 - self.momentum)

        z_t = z.permute(0, 2, 1)  # b * k * n
        x = mu.matmul(z_t)  # b * c * n
        x = x.view(b, c, h, w)  # b * c * h * w
        x = self.relu(x)

        # The second 1x1 conv
        x = self.conv2(x)
        x = x + idn
        x = self.relu(x)

        return x

class EMAU1(nn.Module):
    '''The Expectation-Maximization Attention Unit (EMAU).

    Arguments:
        c (int): The input and output channel number.
        k (int): The number of the bases.
        stage_num (int): The iteration number for EM.
    '''
    def __init__(self, channel, k, stage_num=3, norm_layer=nn.BatchNorm2d):
        super(EMAU1, self).__init__()
        self.stage_num = stage_num

        mu = torch.Tensor(1, channel, k)
        mu.normal_(0, math.sqrt(2. / k))  # Init with Kaiming Norm.
        mu = self._l2norm(mu, dim=1)
        self.register_buffer('mu', mu)
        self.momentum = 0.9
        # self.file = open('size.txt', 'w')

        self.conv1 = nn.Conv2d(channel, channel, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel, channel, 1, bias=False),
            norm_layer(channel))
        self.relu = nn.ReLU(True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _l2norm(self, inp, dim):
        '''Normlize the inp tensor with l2-norm.

        Returns a tensor where each sub-tensor of input along the given dim is
        normalized such that the 2-norm of the sub-tensor is equal to 1.

        Arguments:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.

        Returns:
            (tensor) The normalized tensor.
        '''
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

    def forward(self, x):
        idn = x
        # The first 1x1 conv
        x = self.conv1(x)

        # The EM Attention
        b, c, h, w = int_size(x)
        n = int(h * w)
        x = x.view(b, c, n)  # b * c * n

        # mu = torch.cat([self.mu], 0)
        mu = self.mu.clone()
        with torch.no_grad():
            x_t = x.permute(0, 2, 1).contiguous()  # b * n * c
            for i in range(self.stage_num):
                # z = torch.bmm(x_t, mu)  # b * n * k
                # z = torch.matmul(x_t, mu)  # b * n * k
                z = x_t.matmul(mu)  # b * n * k
                # z = torch.mm(x_t.squeeze(0), mu.squeeze(0))  # b * n * k
                # z = z.unsqueeze(0)
                z = F.softmax(z, dim=-1)  # b * n * k
                z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
                mu = torch.bmm(x, z_)  # b * c * k
                mu = self._l2norm(mu, dim=1)

        # !!! The moving averaging operation
            if self.training:
                mmu = mu.mean(dim=0, keepdim=True)
                self.mu *= self.momentum
                self.mu += mmu * (1 - self.momentum)

        z_t = z.permute(0, 2, 1).contiguous()  # b * k * n
        x = mu.matmul(z_t)  # b * c * n
        x = x.view(b, c, h, w)  # b * c * h * w
        x = self.relu(x)

        # The second 1x1 conv
        x = self.conv2(x)
        x = x + idn
        x = self.relu(x)

        return x

# EMANet for TensorRT
class EMAU(nn.Module):
    '''The Expectation-Maximization Attention Unit (EMAU).

    Arguments:
        c (int): The input and output channel number.
        k (int): The number of the bases.
        stage_num (int): The iteration number for EM.
    '''
    def __init__(self, channel, k, stage_num=3, norm_layer=nn.BatchNorm2d):
        super(EMAU, self).__init__()
        self.stage_num = stage_num

        mu = torch.Tensor(channel, k)
        mu.normal_(0, math.sqrt(2. / k))  # Init with Kaiming Norm.
        mu = self._l2norm(mu, dim=0)
        self.register_buffer('mu', mu)
        self.momentum = 0.9

        self.conv1 = nn.Conv2d(channel, channel, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel, channel, 1, bias=False),
            norm_layer(channel))
        self.relu = nn.ReLU(True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _l2norm(self, inp, dim):
        '''Normlize the inp tensor with l2-norm.

        Returns a tensor where each sub-tensor of input along the given dim is
        normalized such that the 2-norm of the sub-tensor is equal to 1.

        Arguments:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.

        Returns:
            (tensor) The normalized tensor.
        '''
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

    def forward(self, x):
        idn = x
        # The first 1x1 conv
        x = self.conv1(x)

        # The EM Attention
        b, c, h, w = int_size(x)
        n = int(h * w)
        x = x.view(b, c, n)  # b * c * n

        mu = self.mu
        with torch.no_grad():
            x_t = x.permute(0, 2, 1).contiguous()  # b * n * c
            for i in range(self.stage_num):
                z = torch.matmul(x_t, mu)  # b * n * k
                z = F.softmax(z, dim=2)  # b * n * k
                z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
                mu = torch.bmm(x, z_)  # b * c * k
                mu = self._l2norm(mu, dim=1)

        # !!! The moving averaging operation
            if self.training:
                mmu = mu.mean(dim=0)
                self.mu *= self.momentum
                self.mu += mmu * (1 - self.momentum)

        z_t = z.permute(0, 2, 1).contiguous()  # b * k * n
        x = mu.matmul(z_t)  # b * c * n
        x = x.view(b, c, h, w)  # b * c * h * w
        x = self.relu(x)

        # The second 1x1 conv
        x = self.conv2(x)
        x = x + idn
        x = self.relu(x)

        return x

@ATTENTION.register_module()
class EMA(nn.Module):
    '''The Expectation-Maximization Attention Unit (EMAU).

    Arguments:
        c (int): The input and output channel number.
        k (int): The number of the bases.
        stage_num (int): The iteration number for EM.
    '''
    def __init__(self, in_channels, k, stage_num=3, norm_layer=nn.BatchNorm2d):
        super(EMA, self).__init__()
        self.stage_num = stage_num

        mu = torch.Tensor(in_channels, k)
        mu.normal_(0, math.sqrt(2. / k))  # Init with Kaiming Norm.
        mu = self._l2norm(mu, dim=0)
        self.register_buffer('mu', mu)
        self.momentum = 0.9

        self.conv1 = nn.Conv2d(in_channels, in_channels, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            norm_layer(in_channels))
        self.relu = nn.ReLU(True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _l2norm(self, inp, dim):
        '''Normlize the inp tensor with l2-norm.

        Returns a tensor where each sub-tensor of input along the given dim is
        normalized such that the 2-norm of the sub-tensor is equal to 1.

        Arguments:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.

        Returns:
            (tensor) The normalized tensor.
        '''
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

    def forward(self, x):
        # The first 1x1 conv
        x = self.conv1(x)

        # The EM Attention
        b, c, h, w = int_size(x)
        n = int(h * w)
        x = x.view(b, c, n)  # b * c * n

        mu = self.mu
        with torch.no_grad():
            x_t = x.permute(0, 2, 1).contiguous()  # b * n * c
            for i in range(self.stage_num):
                z = torch.matmul(x_t, mu)  # b * n * k
                z = F.softmax(z, dim=2)  # b * n * k
                z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
                mu = torch.bmm(x, z_)  # b * c * k
                mu = self._l2norm(mu, dim=1)

        # !!! The moving averaging operation
            if self.training:
                mmu = mu.mean(dim=0)
                self.mu *= self.momentum
                self.mu += mmu * (1 - self.momentum)

        z_t = z.permute(0, 2, 1).contiguous()  # b * k * n
        x = mu.matmul(z_t)  # b * c * n
        x = x.view(b, c, h, w)  # b * c * h * w
        x = self.relu(x)

        # The second 1x1 conv
        x = self.conv2(x)

        return x


# ---------------------------------------------------------------------------- #
# ECA-Net: Efficient Channel Attention
# ---------------------------------------------------------------------------- #
@ATTENTION.register_module()
class ECA(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, in_channels, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = int_size(x)

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module, y shape [b, c, 1, 1]
        # y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # y = self.conv(y.view(b, c, 1).permute(0, 2, 1)).permute(0, 2, 1).view(b, c, 1, 1)
        y = self.conv(y.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y

@ATTENTION.register_module()
class AECA(nn.Module):
    """Constructs a Adaptive ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, in_channels, gamma=2, b=1):
        super(AECA, self).__init__()
        # t = int(abs((math.log(channels, 2) + b) / gamma))
        # k_size = t if t % 2 else t + 1
        if in_channels == 64:
            k_size = 3
        elif in_channels == 128:
            k_size = 5
        elif in_channels in [256, 512]:
            k_size = 7
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        # self.conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = int_size(x)

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module, y shape [b, c, 1, 1]
        # y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # y = self.conv(y.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        y = self.conv(y.view(b, c, 1).permute(0, 2, 1)).permute(0, 2, 1).view(b, c, 1, 1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y


# ---------------------------------------------------------------------------- #
# Rotate to Attend: Convolutional Triplet Attention Module
# TripletAttention :: https://github.com/LandskapeAI/triplet-attention
# ---------------------------------------------------------------------------- #
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

class SpatialGate(nn.Module):
    def __init__(self, kernel_size = 3):
        super(SpatialGate, self).__init__()
        self.compress = ChannelPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale

@ATTENTION.register_module()
class TPA(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=3, no_spatial=False):
        super(TPA, self).__init__()
        self.cw = SpatialGate()
        self.hc = SpatialGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.hw = SpatialGate()
    def forward(self, x):
        x_perm1 = x.permute(0,2,1,3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        x_perm2 = x.permute(0,3,2,1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = (1/3)*(x_out + x_out11 + x_out21)
        else:
            x_out = (1/2)*(x_out11 + x_out21)
        return x_out