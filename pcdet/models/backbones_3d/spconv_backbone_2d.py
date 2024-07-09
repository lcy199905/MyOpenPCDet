from functools import partial

import torch.nn as nn

from pcdet.utils.spconv_utils import replace_feature, spconv

import torch
from efficientnet_pytorch.model import MemoryEfficientSwish
from einops import rearrange
import torch.nn.functional as F
from timm.models.layers import DropPath
import torch.nn as nn


class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = True
                 ):
        super().__init__()

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        w1 = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts)  # 大于门限值的设为1，否则保留原值
        w2 = torch.where(reweigts > self.gate_treshold, torch.zeros_like(reweigts), reweigts)  # 大于门限值的设为0，否则保留原值
        x_1 = w1 * x
        x_2 = w2 * x
        y = self.reconstruct(x_1, x_2)
        return y

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''

    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        # up
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        # low
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Split
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        # Fuse
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2

class SparseSRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = True
                 ):
        super().__init__()

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        w1 = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts)  # 大于门限值的设为1，否则保留原值
        w2 = torch.where(reweigts > self.gate_treshold, torch.zeros_like(reweigts), reweigts)  # 大于门限值的设为0，否则保留原值
        x_1 = w1 * x
        x_2 = w2 * x
        y = self.reconstruct(x_1, x_2)
        return y

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)

class SparseCRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''

    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel

        self.squeeze1 = spconv.SubMConv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = spconv.SubMConv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        # up
        self.GWC = spconv.SparseConv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2)
        self.PWC1 = spconv.SubMConv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        # low
        self.PWC2 = spconv.SubMConv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # Split
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        # Fuse
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


class ScConv(nn.Module):
    def __init__(self,
                 op_channel: int,
                 group_num: int = 4,
                 gate_treshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.SRU = SRU(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold)
        self.CRU = CRU(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x

class SparseScConv(nn.Module):
    def __init__(self,
                 op_channel: int,
                 group_num: int = 4,
                 gate_treshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.SRU = SparseSRU(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold)
        self.CRU = SparseCRU(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):
    if conv_type == 'subm':
        conv = spconv.SubMConv2d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv2d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


def post_act_block_dense(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, norm_fn=None):
    m = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation, bias=False),
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


'''-------------一、SE模块-----------------------------'''

class SparseGlobalMaxPool(spconv.SparseModule):
    def forward(self, input):
        assert isinstance(input, spconv.SparseConvTensor)
        features = input.features
        device = features.device
        indices = input.indices
        spatial_shape = input.spatial_shape
        batch_size = input.batch_size
        ndim = len(spatial_shape)
        ksize = spatial_shape

        if not self.subm:
            out_spatial_shape = spconv.ops.get_conv_output_size(
                spatial_shape, ksize, [1] * ndim, [0] * ndim,
                [1] * ndim)
        else:
            out_spatial_shape = spatial_shape
        outids, indice_pairs, indice_pairs_num = spconv.ops.get_indice_pairs(
            indices, batch_size, spatial_shape, ksize, 1,
            0, 1, 0, self.subm)

        out_features = spconv.ops.indice_maxpool(features, indice_pairs.to(device),
                                          indice_pairs_num.to(device),
                                          outids.shape[0])
        out_tensor = spconv.SparseConvTensor(out_features, outids,
                                             out_spatial_shape, batch_size)
        out_tensor.indice_dict = input.indice_dict
        out_tensor.grid = input.grid
        return out_tensor

class SparseGlobalAvgPool(spconv.SparseModule):
    def forward(self, input):
        assert isinstance(input, spconv.SparseConvTensor)
        features = input.features
        device = features.device
        indices = input.indices
        spatial_shape = input.spatial_shape
        batch_size = input.batch_size
        ndim = len(spatial_shape)
        ksize = spatial_shape

        if not self.subm:
            out_spatial_shape = spconv.ops.get_conv_output_size(
                spatial_shape, ksize, [1] * ndim, [0] * ndim,
                [1] * ndim)
        else:
            out_spatial_shape = spatial_shape
        outids, indice_pairs, indice_pairs_num = spconv.ops.get_indice_pairs(
            indices, batch_size, spatial_shape, ksize, 1,
            0, 1, 0, self.subm)

        out_features, _ = spconv.ops.indice_avgpool_implicit_gemm(features, indice_pairs.to(device),
                                          indice_pairs_num.to(device))
        out_tensor = spconv.SparseConvTensor(out_features, outids,
                                             out_spatial_shape, batch_size)
        out_tensor.indice_dict = input.indice_dict
        out_tensor.grid = input.grid
        return out_tensor


class SparseChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16, indice_key=None):
        super(SparseChannelAttention, self).__init__()
        # self.avg_pool = SparseGlobalAvgPool()
        # self.avg_pool = spconv.SparseGlobalAvgPool()
        self.avg_pool = spconv.SparseAvgPool2d(1)
        # self.max_pool = spconv.SparseGlobalMaxPool()
        # self.max_pool = SparseGlobalMaxPool()
        self.max_pool = spconv.SparseMaxPool2d(1)

        self.fc1 = spconv.SubMConv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False, indice_key=indice_key)
        self.relu1 = nn.ReLU()
        self.fc2 = spconv.SubMConv2d(in_planes // ratio, in_planes, 1, bias=False, indice_key=indice_key)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        xa = self.avg_pool(x)
        xa = self.fc1(xa)
        xa = replace_feature(xa, self.relu1(xa.features))
        avg_out = self.fc2(xa)

        x = input
        xm = self.max_pool(x)
        xm = self.fc1(xm)
        xm = replace_feature(xm, self.relu1(xm.features))
        max_out = self.fc2(xm)
        out = avg_out + max_out

        return replace_feature(input, self.sigmoid(out.features) * input.features)


class SparseSpatialAttention(nn.Module):
    def __init__(self, kernel_size=3, indice_key=None):
        super(SparseSpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = spconv.SubMConv2d(2, 1, kernel_size, padding=padding, bias=False, indice_key=indice_key)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        avg_out = torch.mean(x.features, dim=1, keepdim=True)
        avg_out = replace_feature(x, avg_out)

        x = input
        max_out, _ = torch.max(x.features, dim=1, keepdim=True)
        max_out = replace_feature(x, max_out)

        x = replace_feature(x, torch.cat([avg_out.features, max_out.features], dim=1))
        x = self.conv1(x)
        return replace_feature(input, self.sigmoid(x.features) * input.features)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x) * input

class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))
        if self.downsample is not None:
            identity = self.downsample(x)
        # out + iden
        out = replace_feature(out, out.features + identity.features)
        # relu(out)
        out = replace_feature(out, self.relu(out.features))

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None):
        super(BasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=bias)
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=bias)
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class PillarBackBone8x_origin(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.sparse_shape = grid_size[[1, 0]]

        block = post_act_block
        dense_block = post_act_block_dense

        self.conv1 = spconv.SparseSequential(
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408] <- [800, 704]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704] <- [400, 352]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352] <- [200, 176]
            block(128, 256, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv4', conv_type='spconv'),
            block(256, 256, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(256, 256, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        norm_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            # [200, 176] <- [100, 88]
            dense_block(256, 256, 3, norm_fn=norm_fn, stride=2, padding=1),
            dense_block(256, 256, 3, norm_fn=norm_fn, padding=1),
            dense_block(256, 256, 3, norm_fn=norm_fn, padding=1),
        )

        self.num_point_features = 256
        self.backbone_channels = {
            'x_conv1': 32,
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256,
            'x_conv5': 256
        }

    def forward(self, batch_dict):
        pillar_features, pillar_coords = batch_dict['pillar_features'], batch_dict['pillar_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=pillar_features,
            indices=pillar_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x_conv1 = self.conv1(input_sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = x_conv4.dense()
        x_conv5 = self.conv5(x_conv4)

        batch_dict.update({
            'multi_scale_2d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
                'x_conv5': x_conv5,
            }
        })
        batch_dict.update({
            'multi_scale_2d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
                'x_conv5': 16,
            }
        })

        return batch_dict

class PillarBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.sparse_shape = grid_size[[1, 0]]

        block = post_act_block
        dense_block = post_act_block_dense

        self.conv1 = spconv.SparseSequential(
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408] <- [800, 704]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704] <- [400, 352]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352] <- [200, 176]
            block(128, 256, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv4', conv_type='spconv'),
            block(256, 256, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(256, 256, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        norm_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            # [200, 176] <- [100, 88]
            dense_block(256, 256, 3, norm_fn=norm_fn, stride=2, padding=1),
            dense_block(256, 256, 3, norm_fn=norm_fn, padding=1),
            dense_block(256, 256, 3, norm_fn=norm_fn, padding=1),
        )

        self.num_point_features = 256
        self.backbone_channels = {
            'x_conv1': 32,
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256,
            'x_conv5': 256
        }

    def forward(self, batch_dict):
        pillar_features, pillar_coords = batch_dict['pillar_features'], batch_dict['pillar_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=pillar_features,
            indices=pillar_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x_conv1 = self.conv1(input_sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = x_conv4.dense()
        x_conv5 = self.conv5(x_conv4)

        batch_dict.update({
            'multi_scale_2d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
                'x_conv5': x_conv5,
            }
        })
        batch_dict.update({
            'multi_scale_2d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
                'x_conv5': 16,
            }
        })

        return batch_dict


class Residual(nn.Module):
    def __init__(self, channels):
        super(Residual, self).__init__()
        self.conv3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bachNorm = nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv3x3(x)
        x = self.bachNorm(x)
        x = self.relu(x)
        return x


class GLFormer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.inter_channel = channels // 2
        self.global_q = nn.Conv2d(channels, self.inter_channel, 1, bias=True)
        self.global_kv = nn.Conv2d(channels, self.inter_channel * 2, 1, bias=True)
        self.avg_pool = nn.AvgPool2d(1)  # 用soft？

        self.local_qkv = nn.Conv2d(channels, self.inter_channel * 3, 1, bias=True)
        self.conv_dw = nn.Conv2d(self.inter_channel, self.inter_channel, 1, groups=self.inter_channel)

        self.fc = nn.Conv2d(self.inter_channel, self.inter_channel, 1)
        self.swish = MemoryEfficientSwish()
        self.tanh = nn.Tanh()

        self.trans_conv = nn.Conv2d(channels, channels, 1, bias=True)
        self.after_norm = nn.GroupNorm(1, channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        self.fc1 = nn.Conv2d(channels, channels, 1)
        self.dwconv = nn.Conv2d(channels, channels, 1, groups=channels)
        self.fc2 = nn.Conv2d(channels, channels, 1)
        self.attn_drop = nn.Dropout(0.3)
        self.drop = nn.Dropout(0.3)
        self.proj_drop = nn.Dropout(0.3)
        self.drop_path = DropPath(0.2)

    def forward(self, x):
        input_data = x
        x = self.after_norm(x)
        q = self.global_q(x)  # BCHW
        kv = self.global_kv(x)  # BCHW
        global_x_kv = self.avg_pool(kv)

        global_kv = rearrange(global_x_kv, 'b (m c) h w -> m b c h w', m=2)
        global_x_q = q  # bchw
        global_x_k = global_kv[0]  # b, c, h w
        global_x_v = global_kv[1]  # b, c, h, w
        B, C, H, W = global_x_k.size()

        energy = torch.mul(global_x_q, global_x_k)
        attention = self.attn_drop(self.softmax(energy))
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = torch.mul(global_x_v, attention)  # b, c, h, w
        x_r = x_r.view(B, C, H, W)
        qkv = self.local_qkv(x)
        local_qkv = rearrange(qkv, 'b (m c) h w -> m b c h w', m=3)
        local_q = self.conv_dw(local_qkv[0])
        local_k = self.conv_dw(local_qkv[1])
        local_v = self.conv_dw(local_qkv[2])
        attn = torch.mul(local_q, local_k)
        attn = self.fc(self.swish(self.fc(attn)))
        attn = self.attn_drop(self.tanh(torch.mul(attn, (1 ** -0.5))))
        local_r = torch.mul(attn, local_v)

        x = torch.cat([local_r, x_r], dim=1)
        x = self.trans_conv(x)
        x = self.proj_drop(x)
        x = input_data + x

        x = self.after_norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dwconv(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return input_data + self.drop_path(x)


class IAFF(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=128, r=4):
        super(IAFF, self).__init__()
        inter_channels = int(channels // r)

        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # nn.AdaptiveMaxPool2d(1),
            # SoftPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()
        # self.residual = Residual(channels)
        self.relu = nn.ReLU()
        # self.block = Block(64)

    def forward(self, x, r=None):
        xa = x + r

        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + r * (1 - wei)
        #
        xl2 = self.local_att(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + r * (1 - wei2)

        return xo


class IAFF1(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=128, r=4):
        super(IAFF1, self).__init__()
        inter_channels = int(channels // r)

        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()
        self.residual = Residual(channels)
        self.relu = nn.ReLU()
        # self.block = Block(64)

    def forward(self, x, r=None):
        r = self.residual(x)
        xa = x + r

        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + r * (1 - wei)
        #
        xl2 = self.local_att(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + r * (1 - wei2)

        return xo


class MAFF(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=128, r=4):
        super(MAFF, self).__init__()
        inter_channels = int(channels // r)

        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        xa = x

        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * (1 + wei)
        #
        xl2 = self.local_att(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * (1 + wei2)

        return xo

class MultiScaleDWConv(nn.Module):
    def __init__(self, dim, scale=(1, 3, 5, 7)):
        super().__init__()
        self.scale = scale
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(scale)):
            if i == 0:
                channels = dim - dim // len(scale) * (len(scale) - 1)
            else:
                channels = dim // len(scale)
            conv = nn.Conv2d(channels, channels,
                             kernel_size=scale[i],
                             padding=scale[i] // 2,
                             groups=channels)
            self.channels.append(channels)
            self.proj.append(conv)

    def forward(self, x):

        x = torch.split(x, split_size_or_sections=self.channels, dim=1)
        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)
        return x

class MultiScaleDWFFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )
        self.dwconv = MultiScaleDWConv(hidden_dim)
        self.act = nn.GELU()
        self.norm = nn.BatchNorm2d(hidden_dim)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )
    def forward(self, x):
        x = self.fc1(x)

        x = self.dwconv(x) + x
        x = self.norm(self.act(x))

        x = self.fc2(x)

        return x

# ------------------------
# class PillarRes18BackBone8x(nn.Module):
#     def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
#         super().__init__()
#         self.model_cfg = model_cfg
#         norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
#         self.sparse_shape = grid_size[[1, 0]]
#
#         block = post_act_block
#         dense_block = post_act_block_dense
#
#         self.conv1 = spconv.SparseSequential(
#             SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res1'),
#             SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res1'),
#         )
#
#         self.conv2 = spconv.SparseSequential(
#             # [1600, 1408] <- [800, 704]
#             block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
#             SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
#             SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
#         )
#
#         self.conv3 = spconv.SparseSequential(
#             # [800, 704] <- [400, 352]
#             block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
#             SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
#             SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
#         )
#
#         self.conv4 = spconv.SparseSequential(
#             # [400, 352] <- [200, 176]
#             block(128, 256, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv4', conv_type='spconv'),
#             SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'),
#             SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'),
#         )
#
#         norm_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
#         self.conv5 = nn.Sequential(
#             # [200, 176] <- [100, 88]
#             dense_block(256, 256, 3, norm_fn=norm_fn, stride=2, padding=1),
#             BasicBlock(256, 256, norm_fn=norm_fn),
#             BasicBlock(256, 256, norm_fn=norm_fn),
#         )
#
#         self.num_point_features = 256
#         self.backbone_channels = {
#             'x_conv1': 32,
#             'x_conv2': 64,
#             'x_conv3': 128,
#             'x_conv4': 256,
#             'x_conv5': 256
#         }
#
#     def forward(self, batch_dict):
#         pillar_features, pillar_coords = batch_dict['pillar_features'], batch_dict['pillar_coords']
#         batch_size = batch_dict['batch_size']
#         input_sp_tensor = spconv.SparseConvTensor(
#             features=pillar_features,
#             indices=pillar_coords.int(),
#             spatial_shape=self.sparse_shape,
#             batch_size=batch_size
#         )
#
#         x_conv1 = self.conv1(input_sp_tensor)
#         x_conv2 = self.conv2(x_conv1)
#         x_conv3 = self.conv3(x_conv2)
#         x_conv4 = self.conv4(x_conv3)
#         x_conv4 = x_conv4.dense()
#         x_conv5 = self.conv5(x_conv4)
#
#         batch_dict.update({
#             'multi_scale_2d_features': {
#                 'x_conv1': x_conv1,
#                 'x_conv2': x_conv2,
#                 'x_conv3': x_conv3,
#                 'x_conv4': x_conv4,
#                 'x_conv5': x_conv5,
#             }
#         })
#         batch_dict.update({
#             'multi_scale_2d_strides': {
#                 'x_conv1': 1,
#                 'x_conv2': 2,
#                 'x_conv3': 4,
#                 'x_conv4': 8,
#                 'x_conv5': 16,
#             }
#         })
#
#         return batch_dict

class PillarRes18BackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.sparse_shape = grid_size[[1, 0]]

        block = post_act_block
        dense_block = post_act_block_dense

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res1'),
        )
        self.ca32_sa = nn.Sequential(
            SparseChannelAttention(32, 16, indice_key='ca1'),
            SparseSpatialAttention(3, indice_key='sa1')
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408] <- [800, 704]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
        )

        self.ca64_sa = nn.Sequential(
            SparseChannelAttention(64,16,  indice_key='ca2'),
            SparseSpatialAttention(3, indice_key='sa2')
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704] <- [400, 352]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
        )
        self.ca128_sa = nn.Sequential(
            SparseChannelAttention(128, 16, indice_key='ca3'),
            SparseSpatialAttention(3, indice_key='sa3')
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352] <- [200, 176]
            block(128, 256, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'),
        )
        # self.ca256_sa = nn.Sequential(
        #     SparseChannelAttention(256, indice_key='ca4'),
        #     SparseSpatialAttention(3, indice_key='sa4')
        # )

        norm_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            # [200, 176] <- [100, 88]
            dense_block(256, 256, 3, norm_fn=norm_fn, stride=2, padding=1),
            BasicBlock(256, 256, norm_fn=norm_fn),
            BasicBlock(256, 256, norm_fn=norm_fn),
        )

        self.dense_ca256 = ChannelAttention(256)
        self.dense_sa = SpatialAttention(3)

        self.num_point_features = 256
        self.backbone_channels = {
            'x_conv1': 32,
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256,
            'x_conv5': 256
        }

    def forward(self, batch_dict):
        pillar_features, pillar_coords = batch_dict['pillar_features'], batch_dict['pillar_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=pillar_features,
            indices=pillar_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x_conv1 = self.conv1(input_sp_tensor)
        x_conv1 = self.ca32_sa(x_conv1)

        x_conv2 = self.conv2(x_conv1)
        x_conv2 = self.ca64_sa(x_conv2)

        x_conv3 = self.conv3(x_conv2)
        x_conv3 = self.ca128_sa(x_conv3)

        x_conv4 = self.conv4(x_conv3)
        x_conv4 = x_conv4.dense()

        x_conv5 = self.dense_sa(self.dense_ca256(x_conv4))
        x_conv5 = self.conv5(x_conv5)

        batch_dict.update({
            'multi_scale_2d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
                'x_conv5': x_conv5,
            }
        })
        batch_dict.update({
            'multi_scale_2d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
                'x_conv5': 16,
            }
        })

        return batch_dict

class PillarRes18BackBone8xM(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.sparse_shape = grid_size[[1, 0]]

        block = post_act_block
        dense_block = post_act_block_dense

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res1'),
        )
        self.ca64_sa = nn.Sequential(
            SparseChannelAttention(64, 16, indice_key='ca1'),
            SparseSpatialAttention(3, indice_key='sa1')
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408] <- [800, 704]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res2'),
        )

        self.ca128_sa = nn.Sequential(
            SparseChannelAttention(128, 16,  indice_key='ca2'),
            SparseSpatialAttention(3, indice_key='sa2')
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704] <- [400, 352]
            block(128, 256, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res3'),
        )
        self.ca256_sa = nn.Sequential(
            SparseChannelAttention(256, 16, indice_key='ca3'),
            SparseSpatialAttention(3, indice_key='sa3')
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352] <- [200, 176]
            block(256, 384, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(384, 384, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(384, 384, norm_fn=norm_fn, indice_key='res4'),
        )
        # self.ca256_sa_2 = nn.Sequential(
        #     SparseChannelAttention(256, indice_key='ca4'),
        #     SparseSpatialAttention(3, indice_key='sa4')
        # )

        norm_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            # [200, 176] <- [100, 88]
            dense_block(384, 384, 3, norm_fn=norm_fn, stride=2, padding=1),
            BasicBlock(384, 384, norm_fn=norm_fn),
            BasicBlock(384, 384, norm_fn=norm_fn),
        )

        self.dense_ca256 = ChannelAttention(384)
        self.dense_sa = SpatialAttention(3)

        self.num_point_features = 256
        self.backbone_channels = {
            'x_conv1': 64,
            'x_conv2': 128,
            'x_conv3': 256,
            'x_conv4': 384,
            'x_conv5': 384
        }

    def forward(self, batch_dict):
        pillar_features, pillar_coords = batch_dict['pillar_features'], batch_dict['pillar_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=pillar_features,
            indices=pillar_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x_conv1 = self.conv1(input_sp_tensor)
        x_conv1 = self.ca64_sa(x_conv1)

        x_conv2 = self.conv2(x_conv1)
        x_conv2 = self.ca128_sa(x_conv2)

        x_conv3 = self.conv3(x_conv2)
        x_conv3 = self.ca256_sa(x_conv3)

        x_conv4 = self.conv4(x_conv3)
        x_conv4 = x_conv4.dense()

        x_conv5 = self.dense_sa(self.dense_ca256(x_conv4))
        x_conv5 = self.conv5(x_conv5)

        batch_dict.update({
            'multi_scale_2d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
                'x_conv5': x_conv5,
            }
        })
        batch_dict.update({
            'multi_scale_2d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
                'x_conv5': 16,
            }
        })

        return batch_dict

class PillarRes34BackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.sparse_shape = grid_size[[1, 0]]

        block = post_act_block
        dense_block = post_act_block_dense

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res1'),
        )
        self.ca32_sa = nn.Sequential(
            SparseChannelAttention(32, 16, indice_key='ca1'),
            SparseSpatialAttention(3, indice_key='sa1')
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408] <- [800, 704]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
        )

        self.ca64_sa = nn.Sequential(
            SparseChannelAttention(64, 16, indice_key='ca2'),
            SparseSpatialAttention(3, indice_key='sa2')
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704] <- [400, 352]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
        )
        self.ca128_sa = nn.Sequential(
            SparseChannelAttention(128, 16, indice_key='ca3'),
            SparseSpatialAttention(3, indice_key='sa3')
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352] <- [200, 176]
            block(128, 256, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'),
        )

        norm_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            # [200, 176] <- [100, 88]
            dense_block(256, 256, 3, norm_fn=norm_fn, stride=2, padding=1),
            BasicBlock(256, 256, norm_fn=norm_fn),
            BasicBlock(256, 256, norm_fn=norm_fn),
        )

        self.dense_ca256 = ChannelAttention(256)
        self.dense_sa = SpatialAttention(3)

        self.num_point_features = 256
        self.backbone_channels = {
            'x_conv1': 32,
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256,
            'x_conv5': 256
        }

    def forward(self, batch_dict):
        pillar_features, pillar_coords = batch_dict['pillar_features'], batch_dict['pillar_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=pillar_features,
            indices=pillar_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x_conv1 = self.conv1(input_sp_tensor)
        x_conv1 = self.ca32_sa(x_conv1)

        x_conv2 = self.conv2(x_conv1)
        x_conv2 = self.ca64_sa(x_conv2)

        x_conv3 = self.conv3(x_conv2)
        x_conv3 = self.ca128_sa(x_conv3)

        x_conv4 = self.conv4(x_conv3)
        x_conv4 = x_conv4.dense()

        x_conv5 = self.dense_sa(self.dense_ca256(x_conv4))
        x_conv5 = self.conv5(x_conv5)

        batch_dict.update({
            'multi_scale_2d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
                'x_conv5': x_conv5,
            }
        })
        batch_dict.update({
            'multi_scale_2d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
                'x_conv5': 16,
            }
        })

        return batch_dict

class ConvBNReLU1d(nn.Module):
    def __init__(self, in_channels, inter_channels, kernel_size):
        super(ConvBNReLU1d, self).__init__()
        self.conv = nn.Conv2d(in_channels, inter_channels, kernel_size=kernel_size)
        self.norm = nn.BatchNorm2d(inter_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)  # 这里N=4与原文一致
        self.inter_channel = inter_channels
        self.conv1 = ConvBNReLU1d(in_channels, inter_channels, 1, )  # 四个1x1卷积用来减小channel为原来的1/N
        self.conv2 = ConvBNReLU1d(in_channels, inter_channels, 1, )
        self.conv3 = ConvBNReLU1d(in_channels, inter_channels, 1, )
        self.conv4 = ConvBNReLU1d(in_channels, inter_channels, 1, )
        self.out = ConvBNReLU1d(in_channels * 2, out_channels, 1)  # 最后的1x1卷积缩小为原来的channel

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)  # 自适应的平均池化，目标size分别为1x1,2x2,3x3,6x6
        return avgpool(x)

    def upsample(self, x, size):  # 上采样使用双线性插值
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    # def upsample(self, x, size, channel):  # 上采样使用双线性插值
    #     up = nn.ConvTranspose2d(channel, channel, kernel_size=size, stride=size, bias=False).cuda()
    #     norm = nn.BatchNorm2d(channel).cuda()
    #     relu = nn.ReLU().cuda()
    #     return relu(norm(up(x)))

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 4)), size)
        feat4 = self.upsample(self.conv3(self.pool(x, 8)), size)
        # feat1 = self.upsample(self.conv1(self.pool(x, 1)), size, self.inter_channel)
        # feat2 = self.upsample(self.conv2(self.pool(x, 2)), [x // 2 for x in size], self.inter_channel)
        # feat3 = self.upsample(self.conv3(self.pool(x, 4)), [x // 4 for x in size], self.inter_channel)
        # feat4 = self.upsample(self.conv4(self.pool(x, 4)), [x // 4 for x in size], self.inter_channel)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)  # concat 四个池化的结果
        x = self.out(x)
        return x


class PillarRes18BackBone8xGL(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.sparse_shape = grid_size[[1, 0]]

        block = post_act_block
        dense_block = post_act_block_dense

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408] <- [800, 704]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704] <- [400, 352]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352] <- [200, 176]
            block(128, 256, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'),
        )

        norm_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        # self.gl_former = GLFormer(256)
        # self.down = dense_block(256, 256, 3, norm_fn=norm_fn, stride=2, padding=1)
        self.conv5 = nn.Sequential(
            # [200, 176] <- [100, 88]
            dense_block(256, 256, 3, norm_fn=norm_fn, stride=2, padding=1),
            BasicBlock(256, 256, norm_fn=norm_fn),
            BasicBlock(256, 256, norm_fn=norm_fn),
        )

        self.num_point_features = 256
        self.backbone_channels = {
            'x_conv1': 32,
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256,
            'x_conv5': 256
        }
        self.iAFF = IAFF(256, 4)
        self.iAFF2 = IAFF(128, 4)
        self.ppm = PyramidPooling(256, 256)
        # self.ca_256 = ChannelAttention(256)
        # self.ca_128 = ChannelAttention(128)
        # self.sa = SpatialAttention()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 2, stride=2, bias=False),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )

    def forward(self, batch_dict):
        pillar_features, pillar_coords = batch_dict['pillar_features'], batch_dict['pillar_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=pillar_features,
            indices=pillar_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x_conv1 = self.conv1(input_sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        # g_conv3 = x_conv3.dense()
        # g_conv3 = self.ca_128(g_conv3) * g_conv3
        # g_conv3 = self.sa(g_conv3) * g_conv3

        x_conv4 = self.conv4(x_conv3)
        x_conv4 = x_conv4.dense()
        # gl_conv5 = self.gl_former(x_conv4)

        # x_conv4 = self.ca_256(x_conv4) * x_conv4
        # x_conv4 = self.sa(x_conv4) * x_conv4
        x_conv5 = self.conv5(x_conv4)
        # x_conv5 = self.ca_256(x_conv5) * x_conv5
        # x_conv5 = self.sa(x_conv5) * x_conv5
        x_conv5 = self.ppm(x_conv5)
        # gl_conv5 = self.iAFF(x_conv5, gl_conv5)
        up_conv4 = self.up(x_conv5)
        up_conv4 = self.iAFF(x_conv4, up_conv4)
        up_conv3 = self.up2(up_conv4)
        up_conv3 = self.iAFF2(x_conv3.dense(), up_conv3)

        batch_dict.update({
            'multi_scale_2d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': up_conv3,
                'x_conv4': up_conv4,
                'x_conv5': x_conv5,
            }
        })
        # x_conv1 = self.conv1(input_sp_tensor)
        # x_conv2 = self.conv2(x_conv1)
        # x_conv3 = self.conv3(x_conv2)
        # x_conv4 = self.conv4(x_conv3)
        # x_conv4 = x_conv4.dense()
        # gl_conv4 = self.gl_former(x_conv4)
        #
        # x_conv5 = self.conv5(gl_conv4)
        # gl_conv5 = self.ppm(x_conv5)
        # gl_conv5 = self.iAFF(x_conv5, gl_conv5)

        # batch_dict.update({
        #     'multi_scale_2d_features': {
        #         'x_conv1': x_conv1,
        #         'x_conv2': x_conv2,
        #         'x_conv3': x_conv3,
        #         'x_conv4': gl_conv4,
        #         'x_conv5': gl_conv5,
        #     }
        # })
        batch_dict.update({
            'multi_scale_2d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
                'x_conv5': 16,
            }
        })

        return batch_dict