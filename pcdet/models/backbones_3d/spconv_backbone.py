from functools import partial

import torch.nn as nn

from ...utils.spconv_utils import replace_feature, spconv
import torch


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):
    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, bias=None, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        if bias is None:
            bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
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

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class SparseChannelAttention2D(nn.Module):
    def __init__(self, in_planes, ratio=16, indice_key=None):
        super(SparseChannelAttention2D, self).__init__()
        # self.avg_pool = SparseGlobalAvgPool()
        # self.avg_pool = spconv.SparseGlobalAvgPool()
        self.avg_pool = spconv.SparseAvgPool2d(1)
        # self.max_pool = spconv.SparseGlobalMaxPool()
        # self.max_pool = SparseGlobalMaxPool()
        self.max_pool = spconv.SparseMaxPool2d(1)

        self.fc1 = spconv.SubMConv2d(in_planes, in_planes // 16, kernel_size=1, bias=False, indice_key=indice_key)
        self.relu1 = nn.ReLU()
        self.fc2 = spconv.SubMConv2d(in_planes // 16, in_planes, 1, bias=False, indice_key=indice_key)

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


class SparseSpatialAttention2D(nn.Module):
    def __init__(self, kernel_size=7, indice_key=None):
        super(SparseSpatialAttention2D, self).__init__()

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

class SparseChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16, indice_key=None):
        super(SparseChannelAttention, self).__init__()
        self.avg_pool = spconv.SparseAvgPool3d(1)
        self.max_pool = spconv.SparseMaxPool3d(1)

        # self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.fc1 = spconv.SubMConv3d(in_planes, in_planes // ratio, kernel_size=1, bias=False, indice_key=indice_key)
        self.relu1 = nn.ReLU()
        self.fc2 = spconv.SubMConv3d(in_planes // ratio, in_planes, 1, bias=False, indice_key=indice_key)

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
    def __init__(self, kernel_size=7, indice_key=None):
        super(SparseSpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = spconv.SubMConv3d(2, 1, kernel_size, padding=padding, bias=False, indice_key=indice_key)
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

class VoxelBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.ca16_sa = nn.Sequential(
            SparseChannelAttention(16, indice_key='ca1'),
            SparseSpatialAttention(3, indice_key='sa1')
        )

        self.ca32_sa = nn.Sequential(
            SparseChannelAttention(32, indice_key='ca2'),
            SparseSpatialAttention(3, indice_key='sa2')
        )

        self.ca64_sa = nn.Sequential(
            SparseChannelAttention(64, indice_key='ca3'),
            SparseSpatialAttention(3, indice_key='sa3')
        )

        self.ca64_sa_4 = nn.Sequential(
            SparseChannelAttention(64, indice_key='ca4'),
            SparseSpatialAttention(3, indice_key='sa4')
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        # x_conv1 = self.conv1(x)
        # x_conv2 = self.conv2(x_conv1)
        # x_conv3 = self.conv3(x_conv2)
        # x_conv4 = self.conv4(x_conv3)
        x_conv1 = self.conv1(x)
        x_conv1 = self.ca16_sa(x_conv1)
        x_conv2 = self.conv2(x_conv1)
        x_conv2 = self.ca32_sa(x_conv2)
        x_conv3 = self.conv3(x_conv2)
        x_conv3 = self.ca64_sa(x_conv3)
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = self.ca64_sa_4(x_conv4)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict


class VoxelResBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        use_bias = self.model_cfg.get('USE_BIAS', None)
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.ca16_sa = nn.Sequential(
            SparseChannelAttention(16, 16, indice_key='ca1'),
            SparseSpatialAttention(3, indice_key='sa1')
        )

        self.ca32_sa = nn.Sequential(
            SparseChannelAttention(32, 16, indice_key='ca2'),
            SparseSpatialAttention(3, indice_key='sa2')
        )

        self.ca64_sa = nn.Sequential(
            SparseChannelAttention(64, 16, indice_key='ca3'),
            SparseSpatialAttention(3, indice_key='sa3')
        )

        self.ca128_sa = nn.Sequential(
            SparseChannelAttention(128, 16, indice_key='ca4'),
            SparseSpatialAttention(3, indice_key='sa4')
        )

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, bias=use_bias, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, bias=use_bias, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, bias=use_bias, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, bias=use_bias, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        # x_conv1 = self.conv1(x)
        # x_conv2 = self.conv2(x_conv1)
        # x_conv3 = self.conv3(x_conv2)
        # x_conv4 = self.conv4(x_conv3)
        x_conv1 = self.conv1(x)
        x_conv1 = self.ca16_sa(x_conv1)
        x_conv2 = self.conv2(x_conv1)
        x_conv2 = self.ca32_sa(x_conv2)
        x_conv3 = self.conv3(x_conv2)
        x_conv3 = self.ca64_sa(x_conv3)
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = self.ca128_sa(x_conv4)
        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)
        out = self.ca128_sa(out)


        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict


class VoxelResBackBone8x_V2(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        spconv_kernel_sizes = model_cfg.get('SPCONV_KERNEL_SIZES', [3, 3, 3, 3])
        channels = model_cfg.get('CHANNELS', [16, 32, 64, 128, 128])
        out_channel = model_cfg.get('OUT_CHANNEL', 128)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, channels[0], 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(channels[0]),
            nn.ReLU(),
        )
        block = post_act_block

        self.ca16_sa = nn.Sequential(
            SparseChannelAttention(16, indice_key='ca1'),
            SparseSpatialAttention(3, indice_key='sa1')
        )

        self.ca32_sa = nn.Sequential(
            SparseChannelAttention(32, indice_key='ca2'),
            SparseSpatialAttention(3, indice_key='sa2')
        )

        self.ca64_sa = nn.Sequential(
            SparseChannelAttention(64, indice_key='ca3'),
            SparseSpatialAttention(3, indice_key='sa3')
        )

        self.ca128_sa = nn.Sequential(
            SparseChannelAttention(128, indice_key='ca4'),
            SparseSpatialAttention(3, indice_key='sa4')
        )
        self.ca128_sa_2d = nn.Sequential(
            SparseChannelAttention2D(128, indice_key='ca5'),
            SparseSpatialAttention2D(3, indice_key='sa5')
        )

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(channels[0], channels[1], spconv_kernel_sizes[0], norm_fn=norm_fn, stride=2,
                  padding=int(spconv_kernel_sizes[0] // 2), indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(channels[1], channels[2], spconv_kernel_sizes[1], norm_fn=norm_fn, stride=2,
                  padding=int(spconv_kernel_sizes[1] // 2), indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 6]
            block(channels[2], channels[3], spconv_kernel_sizes[2], norm_fn=norm_fn, stride=2,
                  padding=int(spconv_kernel_sizes[2] // 2), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'),
        )

        self.conv5 = spconv.SparseSequential(
            # [200, 176, 6] <- [100, 88, 3]
            block(channels[3], channels[4], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2,
                  padding=int(spconv_kernel_sizes[3] // 2), indice_key='spconv5', conv_type='spconv'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res5'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res5'),
        )

        self.conv6 = spconv.SparseSequential(
            # [200, 176, 6] <- [100, 88, 3]
            block(channels[4], channels[4], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2,
                  padding=int(spconv_kernel_sizes[3] // 2), indice_key='spconv6', conv_type='spconv'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res6'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res6'),
        )
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv2d(channels[3], out_channel, 3, stride=1, padding=1, bias=False,
                                indice_key='spconv_down2'),
            norm_fn(out_channel),
            nn.ReLU(),
        )

        self.shared_conv = spconv.SparseSequential(
            spconv.SubMConv2d(out_channel, out_channel, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(True),
        )
        # last_pad = 0
        # last_pad = self.model_cfg.get('last_pad', last_pad)
        # self.conv_out = spconv.SparseSequential(
        #     # [200, 150, 5] -> [200, 150, 2]
        #     spconv.SparseConv3d(channels[3], out_channel, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
        #                         bias=False, indice_key='spconv_down2'),
        #     norm_fn(128),
        #     nn.ReLU(),
        # )
        self.forward_ret_dict = {}
        self.num_point_features = out_channel
        self.backbone_channels = {
            'x_conv1': channels[0],
            'x_conv2': channels[1],
            'x_conv3': channels[2],
            'x_conv4': channels[3]
        }
    def bev_out(self, x_conv):
        features_cat = x_conv.features
        indices_cat = x_conv.indices[:, [0, 2, 3]]
        spatial_shape = x_conv.spatial_shape[1:]

        indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True)
        features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
        features_unique.index_add_(0, _inv, features_cat)

        x_out = spconv.SparseConvTensor(
            features=features_unique,
            indices=indices_unique,
            spatial_shape=spatial_shape,
            batch_size=x_conv.batch_size
        )
        return x_out
    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv1 = self.ca16_sa(x_conv1)

        x_conv2 = self.conv2(x_conv1)
        x_conv2 = self.ca32_sa(x_conv2)

        x_conv3 = self.conv3(x_conv2)
        x_conv3 = self.ca64_sa(x_conv3)

        x_conv4 = self.conv4(x_conv3)
        x_conv4 = self.ca128_sa(x_conv4)
        x_conv5 = self.conv5(x_conv4)
        x_conv5 = self.ca128_sa(x_conv5)
        x_conv6 = self.conv6(x_conv5)
        x_conv6 = self.ca128_sa(x_conv6)

        x_conv5.indices[:, 1:] *= 2
        x_conv6.indices[:, 1:] *= 4
        x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv5.features, x_conv6.features]))
        x_conv4.indices = torch.cat([x_conv4.indices, x_conv5.indices, x_conv6.indices])



        out = self.bev_out(x_conv4)

        out = self.conv_out(out)
        out = self.ca128_sa_2d(out)

        out = self.shared_conv(out)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict